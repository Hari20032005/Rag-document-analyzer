/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        slatebrand: {
          50: '#f6f7fb',
          100: '#edf0f8',
          900: '#18212f',
        },
      },
      boxShadow: {
        soft: '0 12px 40px rgba(24, 33, 47, 0.12)',
      },
    },
  },
  plugins: [],
};
