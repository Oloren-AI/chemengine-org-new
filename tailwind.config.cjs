/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	darkMode: "media", // or 'media' or 'class'
	theme: {
	  extend: {},
	  fontFamily: {
		sans: ['"Inter"', "serif"],
		body: ["Inter", "serif"],
		display: ["Inter", "serif"],
	  },
	},
	variants: {
	  extend: {},
	},
	plugins: [
	  require("daisyui"),
	  require('@tailwindcss/typography'),
	  function ({ addVariant }) {
		addVariant("child", "& > *");
		addVariant("child-hover", "& > *:hover");
	  },
	],
	daisyui: {
	  styled: true,
	  themes: [
		{
		  light: {
			primary: "#0D2C69",
			secondary: "#475F8D",
			accent: "#48DB72",
			neutral: "#ECECEC",
			"base-100": "#FFFFFF",
			info: "#FFFFFF",
			success: "#48DB72",
			warning: "#FBBD23",
			error: "#F87272",
		  },
		},
	  ],
	},
}
