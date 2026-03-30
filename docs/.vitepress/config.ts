import { defineConfig } from "vitepress";

const siteTitle = "ComfyUI-LTXLongAudio";
const siteDescription = "Native LTX custom nodes for long-audio workflows in ComfyUI.";
const siteOrigin = "https://sunwood-ai-labs.github.io";
const siteBase = "/ComfyUI-LTXLongAudio/";
const siteUrl = new URL(siteBase, siteOrigin).toString();
const ogImageUrl = new URL("ogp.png", siteUrl).toString();
const repoUrl = "https://github.com/Sunwood-ai-labs/ComfyUI-LTXLongAudio";
const footer = {
  message: "Built for practical long-audio ComfyUI workflows.",
  copyright: "Copyright © 2026 Sunwood AI Labs",
};

function toPagePath(page: string): string {
  if (page === "index.md") return "/";
  if (page.endsWith("/index.md")) return `/${page.replace(/\/index\.md$/, "")}/`;
  return `/${page.replace(/\.md$/, "")}`;
}

function toAbsoluteUrl(path: string): string {
  return new URL(path.replace(/^\/+/, ""), siteUrl).toString();
}

export default defineConfig({
  title: siteTitle,
  description: siteDescription,
  lang: "en-US",
  base: siteBase,
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: `${siteBase}favicon.svg` }],
    ["meta", { name: "theme-color", content: "#0B67D1" }],
  ],
  sitemap: {
    hostname: siteUrl,
  },
  transformHead({ page, title, description }) {
    const pageUrl = toAbsoluteUrl(toPagePath(page));
    const locale = page.startsWith("ja/") ? "ja_JP" : "en_US";

    return [
      ["link", { rel: "canonical", href: pageUrl }],
      ["meta", { property: "og:type", content: "website" }],
      ["meta", { property: "og:site_name", content: siteTitle }],
      ["meta", { property: "og:locale", content: locale }],
      ["meta", { property: "og:title", content: title }],
      ["meta", { property: "og:description", content: description }],
      ["meta", { property: "og:url", content: pageUrl }],
      ["meta", { property: "og:image", content: ogImageUrl }],
      ["meta", { property: "og:image:type", content: "image/png" }],
      ["meta", { property: "og:image:alt", content: "ComfyUI-LTXLongAudio social card" }],
      ["meta", { name: "twitter:card", content: "summary_large_image" }],
      ["meta", { name: "twitter:title", content: title }],
      ["meta", { name: "twitter:description", content: description }],
      ["meta", { name: "twitter:image", content: ogImageUrl }],
    ];
  },
  themeConfig: {
    search: {
      provider: "local",
    },
    socialLinks: [{ icon: "github", link: repoUrl }],
    footer,
  },
  locales: {
    root: {
      label: "English",
      lang: "en-US",
      title: siteTitle,
      description: siteDescription,
      themeConfig: {
        logo: "/logo.svg",
        nav: [
          { text: "Home", link: "/" },
          { text: "Guide", link: "/guide/getting-started" },
          { text: "GitHub", link: repoUrl },
        ],
        sidebar: [
          {
            text: "Guide",
            items: [
              { text: "Getting Started", link: "/guide/getting-started" },
              { text: "Usage", link: "/guide/usage" },
              { text: "Architecture", link: "/guide/architecture" },
              { text: "Troubleshooting", link: "/guide/troubleshooting" },
            ],
          },
        ],
        outline: {
          level: [2, 3],
          label: "On this page",
        },
        editLink: {
          pattern: `${repoUrl}/edit/main/docs/:path`,
          text: "Edit this page",
        },
        docFooter: {
          prev: "Previous page",
          next: "Next page",
        },
        socialLinks: [{ icon: "github", link: repoUrl }],
        footer,
      },
    },
    ja: {
      label: "日本語",
      lang: "ja-JP",
      title: siteTitle,
      description: "長尺音声の ComfyUI ワークフロー向けに設計したネイティブ LTX カスタムノード集です。",
      themeConfig: {
        logo: "/logo.svg",
        nav: [
          { text: "ホーム", link: "/ja/" },
          { text: "ガイド", link: "/ja/guide/getting-started" },
          { text: "GitHub", link: repoUrl },
        ],
        sidebar: [
          {
            text: "ガイド",
            items: [
              { text: "はじめに", link: "/ja/guide/getting-started" },
              { text: "使い方", link: "/ja/guide/usage" },
              { text: "構成", link: "/ja/guide/architecture" },
              { text: "トラブルシュート", link: "/ja/guide/troubleshooting" },
            ],
          },
        ],
        outline: {
          level: [2, 3],
          label: "このページ",
        },
        editLink: {
          pattern: `${repoUrl}/edit/main/docs/:path`,
          text: "このページを編集",
        },
        docFooter: {
          prev: "前のページ",
          next: "次のページ",
        },
        socialLinks: [{ icon: "github", link: repoUrl }],
        footer,
      },
    },
  },
});
