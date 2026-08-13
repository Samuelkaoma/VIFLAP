/// <reference types="vite/client" />

/**
 * Declared environment variables.
 *
 * Vite's own `ImportMetaEnv` carries an `[key: string]: any` index signature, so
 * reading an undeclared variable yields `any` and the base URL of every request
 * this client makes would be untyped. Declaring the one variable the application
 * reads merges with Vite's interface and gives that read a real type.
 *
 * Optional rather than required: the application falls back to loopback, and a
 * required declaration would assert at compile time that a value exists which
 * at runtime may not.
 */
interface ImportMetaEnv {
  readonly VITE_VIFLAP_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
