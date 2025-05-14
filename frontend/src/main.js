import { createApp } from 'vue'
import App from './App.vue'

import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from './components/Dashboard.vue'  // 💡 你要把 Dashboard.vue 放到 /components 目录
import Evaluation from './components/Evaluation.vue'
import AttackLog from './components/AttackLog.vue'
import Settings from './components/Settings.vue'
const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/dashboard',
      name: 'Dashboard',
      component: Dashboard
    },
    {
      path: '/',
      redirect: '/dashboard'
    },
    {
      path: '/evaluation',
      name: 'Evaluation',
      component: Evaluation
    },
    {
      path: '/log',
      name: 'AttackLog',
      component: AttackLog
    },
    {
      path: '/settings',
      name: 'Settings',
      component: Settings
    }
  ]
})

const app = createApp(App)
app.use(router)
app.mount('#app')
