/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Bound to loopback. The interface displays evidence about identified
    // individuals, and a development server listening on every interface is a
    // disclosure waiting for someone to be on the same network.
    host: '127.0.0.1',
  },
  build: { outDir: 'dist', sourcemap: true },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
