<template>
	<div class="log-container">
	  <div class="panel">
		<div class="panel-header">
		  <h2>📝 攻击日志记录</h2>
		</div>
		<table class="log-table">
		  <thead>
			<tr>
			  <th>#</th>
			  <th>时间</th>
			  <th>源 IP</th>
			  <th>类型</th>
			  <th>置信度</th>
			</tr>
		  </thead>
		  <tbody>
			<tr
			  v-for="(entry, index) in logs"
			  :key="entry.id"
			  :class="getSeverityClass(entry.score)"
			>
			  <td>{{ index + 1 }}</td>
			  <td>{{ formatTimestamp(entry.timestamp) }}</td>
			  <td>{{ entry.source_ip }}</td>
			  <td>{{ entry.label }}</td>
			  <td>{{ (entry.score * 100).toFixed(1) }}%</td>
			</tr>
		  </tbody>
		</table>
	  </div>
	</div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue'
  import axios from 'axios'
  
  const logs = ref([])
  
  const fetchLogs = async () => {
	try {
	  const res = await axios.get('/api/logs')
	  logs.value = res.data || []
	} catch (e) {
	  console.error('❌ 获取日志失败', e)
	}
  }
  
  const formatTimestamp = (ts) => {
	const d = new Date(ts)
	return `${d.getFullYear()}-${(d.getMonth() + 1)
	  .toString()
	  .padStart(2, '0')}-${d
	  .getDate()
	  .toString()
	  .padStart(2, '0')} ${d
	  .getHours()
	  .toString()
	  .padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d
	  .getSeconds()
	  .toString()
	  .padStart(2, '0')}`
  }
  
  const getSeverityClass = (score) => {
	if (score >= 0.8) return 'critical'
	if (score >= 0.5) return 'warning'
	return 'info'
  }
  
  onMounted(() => {
	fetchLogs()
  })
  </script>
  
  <style scoped>
  .log-container {
	background-color: #0a1929;
	min-height: 100vh;
	padding: 20px;
	color: #e2e8f0;
	font-family: 'Roboto', sans-serif;
  }
  
  .panel {
	background-color: rgba(26, 32, 44, 0.8);
	border: 1px solid rgba(99, 179, 237, 0.3);
	border-radius: 12px;
	box-shadow: 0 0 20px rgba(66, 153, 225, 0.2);
	padding: 20px;
  }
  
  .panel-header {
	border-bottom: 1px solid rgba(99, 179, 237, 0.2);
	margin-bottom: 20px;
  }
  
  .panel-header h2 {
	font-size: 20px;
	color: #90cdf4;
	margin: 0;
  }
  
  .log-table {
	width: 100%;
	border-collapse: collapse;
	font-size: 14px;
  }
  
  .log-table th,
  .log-table td {
	padding: 10px 12px;
	border-bottom: 1px solid rgba(255, 255, 255, 0.05);
	text-align: left;
  }
  
  .log-table th {
	background-color: rgba(45, 55, 72, 0.8);
	color: #a0aec0;
	font-weight: 600;
  }
  
  .log-table tr.info {
	background-color: rgba(66, 153, 225, 0.08);
  }
  
  .log-table tr.warning {
	background-color: rgba(237, 137, 54, 0.08);
  }
  
  .log-table tr.critical {
	background-color: rgba(245, 101, 101, 0.12);
  }
  </style>
  