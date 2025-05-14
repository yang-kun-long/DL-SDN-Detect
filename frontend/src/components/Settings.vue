<template>
	<div class="settings-container">
		<div class="panel">
			<div class="panel-header">
				<h2>⚙️ 系统设置</h2>
			</div>

			<div class="form-group">
				<label for="threshold">告警阈值（当前：{{ displayThreshold }})</label>
				<input id="threshold" type="range" min="0.3" max="0.9" step="0.01" v-model="threshold" />
				
			</div>



			<div class="form-group">
				<button @click="reloadModel" :disabled="loading">
					{{ loading ? '正在加载模型...' : '🔁 热更新模型' }}
				</button>
				<p v-if="message" class="status-msg">{{ message }}</p>
			</div>
		</div>
	</div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'

const threshold = ref(0.5)
const displayThreshold = computed(() => {
	const val = Number(threshold.value)
	return isNaN(val) ? 'N/A' : val.toFixed(2)
})

const autoBlock = ref(true)
const loading = ref(false)
const message = ref('')

// 初始化：获取当前后端阈值
const fetchThreshold = async () => {
	try {
		const res = await axios.get('/api/get_threshold')
		threshold.value = Number(res.data.threshold) || 0.5
	} catch (err) {
		console.error('❌ 获取阈值失败', err)
		message.value = '⚠️ 获取当前阈值失败，使用默认值'
	}
}

// 更新阈值（点击按钮触发）
const updateThreshold = async () => {
	try {
		const res = await axios.post('/api/set_threshold', {
			value: threshold.value,
		})
		threshold.value = Number(res.data.threshold) // 从后端同步真实保存值
		message.value = `✅ 阈值已保存为 ${res.data.threshold.toFixed(2)}`
	} catch (err) {
		console.error('❌ 阈值保存失败', err)
		message.value = '❌ 阈值保存失败'
	}
}

// 热更新按钮绑定的函数（包含更新阈值）
const reloadModel = async () => {
	loading.value = true
	message.value = ''
	await updateThreshold()
	await new Promise((res) => setTimeout(res, 2000)) // 模拟模型加载延迟
	message.value += ' ✅ 模型已更新于 ' + new Date().toLocaleTimeString()
	loading.value = false
}

onMounted(() => {
	fetchThreshold()
})
</script>

<style scoped>
.slider-value {
	text-align: right;
	font-size: 14px;
	color: #90cdf4;
	margin-top: 4px;
}

.settings-container {
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
	padding: 24px;
	max-width: 600px;
	margin: 0 auto;
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

.form-group {
	margin-bottom: 24px;
}

label {
	display: block;
	font-size: 15px;
	margin-bottom: 8px;
	color: #cbd5e0;
}

input[type='range'] {
	width: 100%;
}

input[type='checkbox'] {
	margin-right: 8px;
}

button {
	background-color: #3182ce;
	color: #fff;
	border: none;
	padding: 10px 20px;
	border-radius: 6px;
	font-size: 14px;
	cursor: pointer;
	transition: background-color 0.2s;
}

button:hover {
	background-color: #2b6cb0;
}

button:disabled {
	background-color: #4a5568;
	cursor: not-allowed;
}

.status-msg {
	margin-top: 10px;
	font-size: 14px;
	color: #68d391;
}
</style>