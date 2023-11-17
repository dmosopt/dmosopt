import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "dmosopt",
  description: "Distributed multi-objective surrogate-assisted optimization",
  base: '/dmosopt/',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/introduction' },
      { text: 'Examples', link: '/examples/zdt1' },
      // { text: 'Reference', link: '/reference/index' }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Introduction', link: '/guide/introduction' },
            { text: 'Configuration', link: '/guide/configuration' },
            { text: 'Optimizers', link: '/guide/optimizers' },
            { text: 'Surrogates', link: '/guide/surrogates' },
            { text: 'Samplers', link: '/guide/sampling' },
            { text: 'Results', link: '/guide/results' },
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Examples',
          items: [
            { text: 'ZDT1', link: '/examples/zdt1' },
          ]
        }
      ],
      '/reference/': [
        {
          text: 'Reference',
          link: '/reference/index'
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/iraikov/dmosopt' }
    ],

    search: {
      provider: 'local'
    }
  }
})
