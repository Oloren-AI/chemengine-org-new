import { defineConfig } from 'astro/config';

// https://astro.build/config
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  site: 'https://oloren-ai.github.io',
  base: '/chemengine-org-new',
  integrations: [tailwind()]
});