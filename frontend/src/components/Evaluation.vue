<template>
	<div class="evaluation-container">
	  <div class="panel">
		<div class="panel-header">
		  <h2>📊 模型评估指标</h2>
		</div>
		<div ref="chartRef" class="chart-box"></div>
	  </div>
	</div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue'
  import * as echarts from 'echarts'
  
  const chartRef = ref(null)
  let chartInstance = null
  
  const initChart = () => {
	chartInstance = echarts.init(chartRef.value)
  
	const option = {
	  backgroundColor: '#0a1929',
	  tooltip: {
		trigger: 'axis'
	  },
	  legend: {
		data: ['准确率', '召回率', 'F1-score'],
		textStyle: { color: '#e2e8f0' }
	  },
	  grid: {
		left: '5%',
		right: '5%',
		bottom: '8%',
		top: '15%',
		containLabel: true
	  },
	  xAxis: {
		type: 'category',
		boundaryGap: false,
		data: ['第1轮', '第2轮', '第3轮', '第4轮', '第5轮', '第6轮'],
		axisLine: { lineStyle: { color: '#63b3ed' } },
		axisLabel: { color: '#cbd5e0' }
	  },
	  yAxis: {
		type: 'value',
		min: 0.7,
		max: 1.0,
		axisLine: { lineStyle: { color: '#63b3ed' } },
		axisLabel: { color: '#cbd5e0' },
		splitLine: { lineStyle: { color: 'rgba(99,179,237,0.1)' } }
	  },
	  series: [
		{
		  name: '准确率',
		  type: 'line',
		  smooth: true,
		  data: [0.92, 0.935, 0.94, 0.946, 0.948, 0.950],
		  color: '#90cdf4'
		},
		{
		  name: '召回率',
		  type: 'line',
		  smooth: true,
		  data: [0.91, 0.928, 0.934, 0.942, 0.944, 0.947],
		  color: '#68d391'
		},
		{
		  name: 'F1-score',
		  type: 'line',
		  smooth: true,
		  data: [0.915, 0.931, 0.937, 0.945, 0.946, 0.949],
		  color: '#f6ad55'
		}
	  ]
	}
  
	chartInstance.setOption(option)
  }
  
  onMounted(() => {
	initChart()
	window.addEventListener('resize', () => chartInstance?.resize())
  })
  </script>
  
  <style scoped>
  .evaluation-container {
	background-color: #0a1929;
	padding: 20px;
	min-height: calc(100vh - 80px);
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
  
  .chart-box {
	width: 100%;
	height: 400px;
  }
  </style>
  