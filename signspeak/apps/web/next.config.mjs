/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    // onnxruntime-web ships .wasm + worker files that webpack must not try to
    // bundle as JS.
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false, crypto: false };
    return config;
  },
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          // onnxruntime-web needs SharedArrayBuffer for multi-threaded WASM,
          // which requires cross-origin isolation. Without these two headers
          // it silently falls back to single-threaded and inference is ~3x
          // slower - no error, just a slow demo.
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
    ];
  },
};
export default nextConfig;
