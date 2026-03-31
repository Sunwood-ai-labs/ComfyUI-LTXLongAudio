import { app } from "../../../scripts/app.js";

const TARGET_NODE = "LTXLoadImageBatchUpload";
const TARGET_WIDGET = "image";

const PATCHED = Symbol("ltxBatchUploadPatched");
const BATCH_SELECTION = Symbol("ltxBatchSelection");
const DISPLAY_VALUE = Symbol("ltxBatchDisplayValue");
const APP_MODE_PATCHED = Symbol("ltxAppModeBatchUploadPatched");

let pendingBatchSelection = null;
let appModeObserver = null;

function normalizeSelection(value) {
  if (value && typeof value === "object" && !Array.isArray(value) && Array.isArray(value.__value__)) {
    return value.__value__.map(String).filter(Boolean);
  }
  if (Array.isArray(value)) {
    return value.map(String).filter(Boolean);
  }
  if (typeof value === "string") {
    return value ? [value] : [];
  }
  if (value == null || value === "") {
    return [];
  }
  return [String(value)];
}

function basename(value) {
  return String(value).split("/").pop();
}

function getSerializableWidgetIndex(node, targetWidgetName) {
  let serializableIndex = 0;
  for (const widget of node.widgets ?? []) {
    if (widget.serialize === false) {
      continue;
    }
    if (widget.name === targetWidgetName) {
      return serializableIndex;
    }
    serializableIndex += 1;
  }
  return -1;
}

function getComboWidget(node) {
  return node.widgets?.find((candidate) => candidate.name === TARGET_WIDGET);
}

function getFirstBatchWidget() {
  const node = app.graph?._nodes?.find((candidate) => candidate.type === TARGET_NODE);
  return node ? getComboWidget(node) : null;
}

function promotePendingSelection(nextValue, widget) {
  const directSelection = normalizeSelection(nextValue);
  const selection = pendingBatchSelection;
  if (!selection?.length || typeof nextValue !== "string") {
    return directSelection;
  }
  if (basename(nextValue) !== basename(selection[0])) {
    return directSelection;
  }
  const availableValues = Array.isArray(widget.options?.values) ? widget.options.values : [];
  const hasEveryUploadedPath = selection.every((candidate) =>
    availableValues.some((option) => basename(option) === basename(candidate))
  );
  if (!hasEveryUploadedPath) {
    return directSelection;
  }
  pendingBatchSelection = null;
  return selection.slice();
}

function commitPendingBatchSelection() {
  const selection = pendingBatchSelection?.slice();
  if (!selection?.length) {
    return;
  }

  const widget = getFirstBatchWidget();
  if (!widget) {
    return;
  }

  const commitWhenReady = () => {
    const availableValues = Array.isArray(widget.options?.values) ? widget.options.values : [];
    const ready = selection.every((candidate) =>
      availableValues.some((option) => basename(option) === basename(candidate))
    );
    if (!ready) {
      requestAnimationFrame(commitWhenReady);
      return;
    }
    setTimeout(() => {
      widget.value = selection;
      pendingBatchSelection = null;
    }, 0);
  };

  commitWhenReady();
}

function installBatchWidgetAdapter(node) {
  if (!node?.widgets?.length) {
    return;
  }

  const widget = getComboWidget(node);
  if (!widget || widget[PATCHED]) {
    return;
  }

  const initialSelection = normalizeSelection(widget.value);

  widget[PATCHED] = true;
  widget[BATCH_SELECTION] = initialSelection;
  widget[DISPLAY_VALUE] = initialSelection[0] ?? (typeof widget.value === "string" ? widget.value : "");

  Object.defineProperty(widget, "value", {
    configurable: true,
    enumerable: true,
    get() {
      return widget[DISPLAY_VALUE];
    },
    set(nextValue) {
      const selection = promotePendingSelection(nextValue, widget);
      widget[BATCH_SELECTION] = selection;
      widget[DISPLAY_VALUE] = selection[0] ?? "";
    },
  });

  const originalCallback = widget.callback;
  widget.callback = function (nextValue) {
    const selection = promotePendingSelection(nextValue, widget);
    if (selection.length > 1) {
      widget.value = selection;
    }
    return originalCallback?.apply(this, arguments);
  };

  widget.serializeValue = () => {
    const selection = widget[BATCH_SELECTION] || [];
    return selection.length > 1 ? [...selection] : widget[DISPLAY_VALUE];
  };

  widget.value = initialSelection.length > 1 ? initialSelection : widget[DISPLAY_VALUE];
}

function appModeImageUploadDescriptors() {
  const graph = app.graph;
  const linearInputs = graph?.extra?.linearData?.inputs ?? [];
  const nodes = graph?._nodes ?? [];
  return linearInputs
    .map(([nodeId, inputName]) => {
      const node = nodes.find((candidate) => candidate.id === nodeId);
      if (!node) {
        return null;
      }
      const hasImageUpload = node.inputs?.some((input) => input.type === "IMAGEUPLOAD");
      if (!hasImageUpload) {
        return null;
      }
      return {
        nodeType: node.type,
        inputName,
      };
    })
    .filter(Boolean);
}

function onAppModeBatchInputChange(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement) || !target.multiple) {
    return;
  }
  const files = Array.from(target.files ?? []);
  pendingBatchSelection = files.length > 1 ? files.map((file) => file.name) : null;
  if (pendingBatchSelection) {
    commitPendingBatchSelection();
  }
}

function syncAppModeBatchUploadInputs() {
  const descriptors = appModeImageUploadDescriptors();
  if (!descriptors.length) {
    return;
  }
  const fileInputs = Array.from(
    document.querySelectorAll('label.relative > input[type="file"][accept^="image/"]')
  );
  if (!fileInputs.length) {
    return;
  }

  descriptors.forEach((descriptor, index) => {
    const fileInput = fileInputs[index];
    if (!(fileInput instanceof HTMLInputElement)) {
      return;
    }
    const isBatchInput = descriptor.nodeType === TARGET_NODE;
    fileInput.multiple = isBatchInput;
    if (fileInput[APP_MODE_PATCHED]) {
      return;
    }
    fileInput[APP_MODE_PATCHED] = true;
    fileInput.addEventListener("change", onAppModeBatchInputChange, true);
  });
}

function ensureAppModeObserver() {
  if (appModeObserver || typeof MutationObserver === "undefined" || !document.body) {
    return;
  }
  appModeObserver = new MutationObserver(() => syncAppModeBatchUploadInputs());
  appModeObserver.observe(document.body, { childList: true, subtree: true });
  syncAppModeBatchUploadInputs();
}

app.registerExtension({
  name: "ComfyUI.LTXLongAudio.MultiImageUpload",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TARGET_NODE) {
      return;
    }

    const originalNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalNodeCreated?.apply(this, arguments);
      installBatchWidgetAdapter(this);
      queueMicrotask(() => syncAppModeBatchUploadInputs());
      return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = originalConfigure?.apply(this, arguments);
      installBatchWidgetAdapter(this);
      queueMicrotask(() => syncAppModeBatchUploadInputs());
      return result;
    };

    const originalSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (serialized) {
      originalSerialize?.apply(this, arguments);
      const widget = getComboWidget(this);
      if (!widget) {
        return;
      }
      const widgetIndex = getSerializableWidgetIndex(this, TARGET_WIDGET);
      const selection = widget[BATCH_SELECTION] || [];
      if (widgetIndex >= 0 && selection.length > 1) {
        serialized.widgets_values ||= [];
        serialized.widgets_values[widgetIndex] = [...selection];
      }
    };
  },

  async nodeCreated(node) {
    if (node.comfyClass === TARGET_NODE) {
      installBatchWidgetAdapter(node);
      queueMicrotask(() => syncAppModeBatchUploadInputs());
    }
  },

  async setup() {
    ensureAppModeObserver();
  },
});
