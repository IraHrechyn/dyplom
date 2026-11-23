export function pythonChildEnv(
  base: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
  return {
    ...base,
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
  }
}
