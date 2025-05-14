<script setup>
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="app-wrapper">
    <header class="nav-bar">
      <h1 class="nav-title">🔒 DDoS 可视化防御系统</h1>
      <nav class="nav-links">
        <RouterLink to="/dashboard" active-class="active-link">Dashboard</RouterLink>
        <RouterLink to="/evaluation" active-class="active-link">模型评估</RouterLink>
        <RouterLink to="/log" active-class="active-link">攻击日志</RouterLink>
        <RouterLink to="/settings" active-class="active-link">系统设置</RouterLink>
      </nav>
    </header>

    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<style scoped>
.app-wrapper {
  background-color: #0a1929;
  min-height: 100vh;
  color: #e2e8f0;
  font-family: 'Roboto', sans-serif;
}

.nav-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 24px;
  background-color: rgba(26, 32, 44, 0.8);
  border-bottom: 1px solid rgba(99, 179, 237, 0.2);
  box-shadow: 0 0 12px rgba(66, 153, 225, 0.2);
  position: sticky;
  top: 0;
  z-index: 999;
}

.nav-title {
  font-size: 20px;
  font-weight: bold;
  color: #90cdf4;
  text-shadow: 0 0 6px rgba(144, 205, 244, 0.4);
  margin: 0;
}

.nav-links {
  display: flex;
  gap: 24px;
}

.nav-links a {
  color: #a0aec0;
  text-decoration: none;
  font-size: 15px;
  position: relative;
  transition: color 0.3s ease;
}

.nav-links a:hover {
  color: #ffffff;
}

.active-link {
  color: #63b3ed;
  font-weight: bold;
}

.main-content {
  padding: 24px;
}
</style>
