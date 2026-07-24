import type { EditorTheme, MarkdownTheme, SelectListTheme } from "@earendil-works/pi-tui";

const ansi = (code: number) => (text: string): string => `\u001b[${code}m${text}\u001b[0m`;

export const colors = {
  cyan: ansi(36),
  green: ansi(32),
  yellow: ansi(33),
  red: ansi(31),
  blue: ansi(34),
  bold: ansi(1),
  dim: ansi(2),
  italic: ansi(3),
  underline: ansi(4),
  strike: ansi(9),
};

const selectList: SelectListTheme = {
  selectedPrefix: (text) => colors.cyan(text),
  selectedText: (text) => colors.bold(text),
  description: (text) => colors.dim(text),
  scrollInfo: (text) => colors.dim(text),
  noMatch: (text) => colors.dim(text),
};

export const editorTheme: EditorTheme = {
  borderColor: colors.cyan,
  selectList,
};

export const markdownTheme: MarkdownTheme = {
  heading: colors.bold,
  link: colors.cyan,
  linkUrl: colors.dim,
  code: colors.yellow,
  codeBlock: colors.green,
  codeBlockBorder: colors.dim,
  quote: colors.dim,
  quoteBorder: colors.blue,
  hr: colors.dim,
  listBullet: colors.cyan,
  bold: colors.bold,
  italic: colors.italic,
  strikethrough: colors.strike,
  underline: colors.underline,
};
