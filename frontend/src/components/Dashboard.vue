<template>
  <div class="dashboard-container">
    <div class="dashboard-header">
      <div class="title-area">
        <h1>SDN 网络安全监控中心</h1>
        <div class="status-indicator">
          系统状态：<span class="status-active">在线</span>
        </div>
      </div>
      <div class="stats-panel">
        <div class="stat-item">
          <div class="stat-value">{{ nodeCount }}</div>
          <div class="stat-label">节点数</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{{ connectionCount }}</div>
          <div class="stat-label">连接数</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{{ alertsCount }}</div>
          <div class="stat-label">告警数</div>
        </div>
      </div>
    </div>

    <div class="dashboard-grid">
      <!-- 网络拓扑图区域 -->
      <!-- 拓扑图区域 -->
      <div class="topology-panel">
        <div class="panel-header">
          <h2>网络拓扑</h2>
          <div class="panel-controls">
            <button @click="zoomIn" class="control-btn">
              <span class="icon">+</span>
            </button>
            <button @click="zoomOut" class="control-btn">
              <span class="icon">-</span>
            </button>
            <button @click="resetView" class="control-btn">
              <span class="icon">↺</span>
            </button>
            <button @click="loadNetworkTopology" class="control-btn">
              <span class="icon">⟳</span>
            </button>
          </div>
        </div>

        <!-- Cytoscape 容器 + loading 遮罩 -->
        <div class="cy-wrapper">
          <div ref="cyRef" class="cy-container"></div>
          <!-- 修复：必须闭合 -->
          <div v-show="loadingTopology" class="loading-overlay">
            <div class="spinner"></div>
            <span style="margin-top: 8px">正在加载拓扑图...</span>
          </div>
        </div>
      </div>

      <!-- 右侧面板区域 -->
      <div class="side-panels">
        <!-- 告警信息区域 -->
        <div class="alert-panel">
          <div class="panel-header">
            <h2>
              <span
                class="pulse-dot"
                :class="{ active: alerts.length > 0 }"
              ></span>
              实时告警监控
            </h2>
            <div class="panel-controls">
              <button @click="toggleSimulation" class="control-btn">
                {{ isSimulating ? "停止模拟" : "启动模拟" }}
              </button>
              <button @click="clearAlerts" class="control-btn">清除</button>
            </div>
          </div>
          <div class="alert-content">
            <div v-if="alerts.length === 0" class="no-alerts">
              <div class="radar-animation"></div>
              <span>系统正常，未检测到威胁</span>
            </div>
            <ul v-else class="alert-list">
              <li
                v-for="alert in alerts"
                :key="alert.id"
                class="alert-item"
                :class="getSeverityClass(alert.score)"
              >
                <div class="alert-header">
                  <span class="alert-timestamp">{{
                    formatTimestamp(alert.timestamp)
                  }}</span>
                  <span class="alert-severity">{{
                    getSeverityLabel(alert.score)
                  }}</span>
                </div>
                <div class="alert-details">
                  <div class="alert-source">
                    源 IP: <span class="highlight">{{ alert.source_ip }}</span>
                  </div>
                  <div class="alert-type">
                    状态: <span class="highlight">{{ alert.label }}</span>
                  </div>
                  <div class="alert-confidence">
                    <span>置信度:</span>
                    <div class="confidence-bar">
                      <div
                        class="confidence-level"
                        :style="{ width: `${alert.score * 100}%` }"
                      ></div>
                    </div>
                    <span class="confidence-value"
                      >{{ (alert.score * 100).toFixed(0) }}%</span
                    >
                  </div>
                </div>
              </li>
            </ul>
          </div>
        </div>

        <!-- 流量分析区域 -->
        <div class="traffic-panel">
          <div class="panel-header">
            <h2>Top-N 流趋势图</h2>
          </div>
          <div class="traffic-content">
            <div class="traffic-chart" ref="trafficChartRef"></div>
            <div class="traffic-stats">
              <div class="traffic-stat">
                <div class="stat-circle incoming">
                  <span>{{ incomingTraffic }}</span>
                  <small>Mbps</small>
                </div>
                <div class="stat-label">sFlow Bytes</div>
              </div>
              <div class="traffic-stat">
                <div class="stat-circle outgoing">
                  <span>{{ outgoingTraffic }}</span>
                  <small>%</small>
                </div>
                <div class="stat-label">系统 CPU 使用率 (%)</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import cytoscape from "cytoscape";
import axios from "axios";
import * as echarts from "echarts";
import { nextTick } from "vue";

// 状态变量
const loadingTopology = ref(true);
const cyRef = ref(null);
const alerts = ref([]);
const trafficChartRef = ref(null);
const trafficData = ref([]);
const incomingTraffic = ref("42.8");
const outgoingTraffic = ref("38.2");
let cy = null;
let trafficChart = null;
const isSimulating = ref(false);
let simulateTimer = null;
const toggleSimulation = () => {
  isSimulating.value = !isSimulating.value;
  if (isSimulating.value) {
    simulateLoop();
  } else {
    simulateRunning.value = false;
  }
};

const simulateRunning = ref(false);

const simulateLoop = async () => {
  if (simulateRunning.value) return;
  simulateRunning.value = true;

  const runOnce = async () => {
    try {
      if (!simulateRunning.value) return;

      const attackRes = await axios.get("api/flow_information");
      if (attackRes.status === 428) {
        console.warn("sFlow 无流，跳过本轮");
        return simulateRunning.value && runOnce();
      }

      const attackData = attackRes.data;
      const predRes = await axios.post("api/predict_from_json", attackData);
      const pred = predRes.data;

      alerts.value.unshift({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        source_ip: pred.source_ip,
        destination_ip: pred.destination_ip,
        label: pred.label === 1 ? "DDoS攻击" : "正常",
        score: pred.score,
      });

      alerts.value = alerts.value.slice(0, 10);

      if (simulateRunning.value) runOnce();
    } catch (err) {
      console.error("模拟攻击失败", err);
      if (simulateRunning.value) {
        setTimeout(runOnce, 3000);
      }
    }
  };

  runOnce();
};

// 计算属性
const nodeCount = computed(() => (cy ? cy.nodes().length : 0));
const connectionCount = computed(() => (cy ? cy.edges().length : 0));
const alertsCount = computed(() => alerts.value.length);

const loadNetworkTopology = async () => {
  loadingTopology.value = true;
  const res = await axios.get("/api/topology");
  const { nodes, links } = res.data;

  const elements = [
    // 加载节点
    ...nodes.map((n) => ({
      data: {
        id: n.id,
        label:
          n.type === "controller"
            ? "SDN控制器"
            : n.type === "switch"
            ? `Switch ${n.id.replace("s", "")}`
            : `Host ${n.id.replace("h", "")}`,
        type: n.type,
        ip: n.ip || null,
        mac: n.mac || null,
      },
    })),
    // 加载连接
    ...links.map((l) => ({
      data: {
        id: `${l.source}-${l.target}`,
        source: l.source,
        target: l.target,
        type: "connection",
      },
    })),
  ];

  // 可选：如果拓扑数据中没有 controller，强制添加
  if (!nodes.find((n) => n.id === "controller")) {
    elements.push({
      data: {
        id: "controller",
        label: "SDN控制器",
        type: "controller",
      },
    });
  }

  // 创建 Cytoscape 实例
  cy = cytoscape({
    container: cyRef.value,
    elements: elements,
    style: [
      {
        selector: "node",
        style: {
          shape: "ellipse",
          width: "60px",
          height: "60px",
          "background-color": "#2a4365",
          "border-width": "2px",
          "border-color": "#63b3ed",
          label: "data(label)",
          color: "#e2e8f0",
          "text-valign": "bottom",
          "text-halign": "center",
          "font-size": "12px",
          "text-outline-width": 1,
          "text-outline-color": "#1a202c",
          "text-margin-y": "8px",
        },
      },
      {
        selector: 'node[type="controller"]',
        style: {
          shape: "hexagon",
          "background-color": "#9f7aea",
          "border-color": "#d6bcfa",
          "border-width": "3px",
          width: "80px",
          height: "80px",
          "background-image":
            'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M4 5h16v10H4z"/><path d="M9 15v2h6v-2h5v5H4v-5z"/></svg>',
          "background-width": "50%",
          "background-height": "50%",
          "background-position-y": "40%",
        },
      },
      {
        selector: 'node[type="switch"]',
        style: {
          shape: "triangle",
          "background-color": "#3182ce",
          "border-color": "#90cdf4",
          width: "65px",
          height: "65px",
          "background-image":
            'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M4 6h16v12H4z"/><path d="M7 9h2v2H7zM7 13h2v2H7zM11 9h2v2h-2zM11 13h2v2h-2zM15 9h2v2h-2zM15 13h2v2h-2z"/></svg>',
          "background-width": "50%",
          "background-height": "50%",
          "background-position-y": "40%",
        },
      },
      {
        selector: 'node[type="host"]',
        style: {
          "background-color": "#38a169",
          "border-color": "#9ae6b4",
          "background-image":
            'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white"><path d="M20 18c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v10c0a1.1.9 2 2 2H0v2h24v-2h-4zM4 6h16v10H4V6z"/></svg>',
          "background-width": "50%",
          "background-height": "50%",
          "background-position-y": "40%",
        },
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": "#63b3ed",
          "target-arrow-color": "#63b3ed",
          "curve-style": "bezier",
          opacity: 0.8,
        },
      },
      {
        selector: 'edge[type="control"]',
        style: {
          width: 3,
          "line-color": "#d6bcfa",
          "line-style": "dashed",
          "line-dash-pattern": [8, 3],
          "target-arrow-shape": "triangle",
          "target-arrow-color": "#d6bcfa",
          opacity: 0.7,
        },
      },
      {
        selector: 'edge[type="link"]',
        style: {
          width: 4,
          "line-color": "#90cdf4",
          "target-arrow-shape": "triangle",
          "source-arrow-shape": "triangle",
          "target-arrow-color": "#90cdf4",
          "source-arrow-color": "#90cdf4",
        },
      },
      {
        selector: 'edge[type="connection"]',
        style: {
          width: 2,
          "line-color": "#9ae6b4",
          "target-arrow-shape": "triangle",
          "target-arrow-color": "#9ae6b4",
        },
      },
      {
        selector: ".highlighted",
        style: {
          "background-color": "#ed8936",
          "border-color": "#fbd38d",
          "border-width": "4px",
          "transition-property": "background-color, border-color, border-width",
          "transition-duration": "0.3s",
        },
      },
      {
        selector: "edge.highlighted",
        style: {
          width: 4,
          "line-color": "#ed8936",
          "target-arrow-color": "#ed8936",
          "source-arrow-color": "#ed8936",
          opacity: 1,
          "transition-property": "line-color, width, opacity",
          "transition-duration": "0.3s",
        },
      },
    ],
    layout: {
      name: "concentric",
      concentric: function (node) {
        if (node.data("type") === "controller") return 3;
        if (node.data("type") === "switch") return 2;
        return 1;
      },
      levelWidth: function () {
        return 1;
      },
      minNodeSpacing: 80,
      animate: true,
    },
  });

  // 鼠标悬停高亮连接线
  cy.on("mouseover", "node", function (e) {
    const node = e.target;
    node.connectedEdges().addClass("highlighted");
  });

  cy.on("mouseout", "node", function (e) {
    const node = e.target;
    node.connectedEdges().removeClass("highlighted");
  });

  animateEdges();

  cy.on("layoutstop", async () => {
    await nextTick();
    loadingTopology.value = false;
    console.log("拓扑图加载完成");
    console.log(loadingTopology.value);
  });
  setTimeout(() => {
    loadingTopology.value = false;
    console.log("loadingTimeout fallback triggered");
  }, 3000);
};

// 边缘动画效果
const animateEdges = () => {
  setInterval(() => {
    cy.edges().forEach((edge) => {
      if (Math.random() > 0.8) {
        const type = edge.data("type");
        const duration = type === "control" ? 800 : 500;

        edge
          .animate({
            style: { "line-color": "#ffffff", opacity: 1 },
            duration: duration,
          })
          .animate({
            style: {
              "line-color":
                type === "control"
                  ? "#d6bcfa"
                  : type === "link"
                  ? "#90cdf4"
                  : "#9ae6b4",
              opacity: type === "control" ? 0.7 : 0.8,
            },
            duration: duration,
          });
      }
    });
  }, 1000);
};

// 模拟数据轮询
// const pollAlerts = async () => {
// 	try {
// 		// 实际使用时替换为实际的API调用
// 		// const res = await axios.get('/status')
// 		// alerts.value = res.data.alerts || []

// 		// 模拟数据
// 		// if (Math.random() > 0.7) {
// 		// 	const newAlert = {
// 		// 		id: Date.now(),
// 		// 		timestamp: new Date().toISOString(),
// 		// 		source_ip: `192.168.1.${Math.floor(Math.random() * 255)}`,
// 		// 		label: getRandomAlertType(),
// 		// 		score: Math.random()
// 		// 	}

// 		// 	// 添加新告警并限制最大数量
// 		// 	alerts.value = [newAlert, ...alerts.value].slice(0, 5);

// 		// 	// 高亮相关节点
// 		// 	highlightCompromisedNode(newAlert);
// 		// }
// 	} catch (e) {
// 		console.error('轮询失败', e)
// 	}
// }

// 随机告警类型
const getRandomAlertType = () => {
  const types = [
    "DDoS攻击",
    "SQL注入尝试",
    "异常流量",
    "端口扫描",
    "未授权访问",
    "ARP欺骗",
    "恶意软件通信",
  ];
  return types[Math.floor(Math.random() * types.length)];
};

// 高亮受影响的节点
const highlightCompromisedNode = (alert) => {
  if (!cy) return;

  // 找到IP匹配的节点或随机选择一个节点
  let targetNode = cy.nodes(`[ip = "${alert.source_ip}"]`);
  if (targetNode.empty()) {
    // 如果没有匹配的IP，随机选择一个主机节点
    const hostNodes = cy.nodes('[type = "host"]');
    if (!hostNodes.empty()) {
      targetNode = hostNodes[Math.floor(Math.random() * hostNodes.length)];
    }
  }

  if (targetNode && !targetNode.empty()) {
    // 高亮节点并在一段时间后恢复
    targetNode.addClass("highlighted");
    setTimeout(() => {
      targetNode.removeClass("highlighted");
    }, 5000);
  }
};

// 初始化流量图表
const initTrafficChart = () => {
  if (!trafficChartRef.value) return;

  trafficChart = echarts.init(trafficChartRef.value);

  const option = {
    grid: {
      left: "3%",
      right: "4%",
      bottom: "3%",
      top: "3%",
      containLabel: true,
    },
    tooltip: {
      trigger: "axis",
      formatter: (params) => {
        return params
          .map((p) => `${p.seriesName}: ${p.data.toFixed(2)} Mbps`)
          .join("<br>");
      },
    },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: [],
      axisLine: {
        lineStyle: {
          color: "#63b3ed",
        },
      },
      axisLabel: {
        color: "#e2e8f0",
      },
    },
    yAxis: {
      type: "value",
      splitLine: {
        lineStyle: {
          color: "rgba(99, 179, 237, 0.2)",
        },
      },
      axisLine: {
        lineStyle: {
          color: "#63b3ed",
        },
      },
      axisLabel: {
        color: "#e2e8f0",
        formatter: "{value} Mbps",
      },
    },
    series: [
      {
        name: "入站流量",
        type: "line",
        stack: "Total",
        data: [],
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              {
                offset: 0,
                color: "rgba(144, 205, 244, 0.8)",
              },
              {
                offset: 1,
                color: "rgba(144, 205, 244, 0.1)",
              },
            ],
          },
        },
        lineStyle: {
          width: 2,
          color: "#90cdf4",
        },
        symbol: "circle",
        symbolSize: 6,
        itemStyle: {
          color: "#90cdf4",
        },
        smooth: true,
      },
      {
        name: "出站流量",
        type: "line",
        stack: "Total",
        data: [],
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              {
                offset: 0,
                color: "rgba(154, 230, 180, 0.8)",
              },
              {
                offset: 1,
                color: "rgba(154, 230, 180, 0.1)",
              },
            ],
          },
        },
        lineStyle: {
          width: 2,
          color: "#9ae6b4",
        },
        symbol: "circle",
        symbolSize: 6,
        itemStyle: {
          color: "#9ae6b4",
        },
        smooth: true,
      },
    ],
  };

  trafficChart.setOption(option);
};

// 更新流量图表
// const updateTrafficChart = () => {
//   if (!trafficChart) return;

//   const now = new Date();
//   const timeStr =
//     now.getHours().toString().padStart(2, "0") +
//     ":" +
//     now.getMinutes().toString().padStart(2, "0") +
//     ":" +
//     now.getSeconds().toString().padStart(2, "0");

//   const incoming = +(Math.random() * 50 + 20).toFixed(1);
//   const outgoing = +(Math.random() * 40 + 15).toFixed(1);

//   incomingTraffic.value = incoming.toFixed(1);
//   outgoingTraffic.value = outgoing.toFixed(1);

//   trafficChart.setOption({
//     xAxis: {
//       data: Array(10)
//         .fill(0)
//         .map((_, i) => (i + 1).toString()),
//     },
//     series: [
//       {
//         data: Array(10)
//           .fill(0)
//           .map(() => +(Math.random() * 50 + 10).toFixed(1)),
//       },
//       {
//         data: Array(10)
//           .fill(0)
//           .map(() => +(Math.random() * 40 + 10).toFixed(1)),
//       },
//     ],
//   });
// };
const updateFlowTrendChart = async () => {
  if (!trafficChart) return;

  try {
    const res = await axios.get("/api/flow_trend", {
      params: {
        keys: "ipsource,ipdestination",
        value: "bytes",
      },
    });

    const { timestamps, series } = res.data;

    const seriesData = Object.entries(series).map(([name, data]) => ({
      name,
      type: "line",
      smooth: true,
      data,
    }));

    trafficChart.setOption({
      xAxis: { data: timestamps },
      legend: { data: Object.keys(series) },
      series: seriesData,
    });
  } catch (e) {
    console.error("flow_trend 获取失败", e);
  }
};

const updateGauges = async () => {
  try {
    const res = await axios.get("/api/traffic_trend");
    const { bps, cpu } = res.data;

    incomingTraffic.value = bps[bps.length - 1].toFixed(1);
    outgoingTraffic.value = cpu.toFixed(1);  // 替代之前的 pps

  } catch (e) {
    console.error("获取仪表盘数据失败", e);
  }
};


// 格式化时间戳
const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
};

// 根据威胁评分获取级别
const getSeverityClass = (score) => {
  if (score >= 0.8) return "critical";
  if (score >= 0.5) return "warning";
  return "info";
};

// 根据威胁评分获取标签
const getSeverityLabel = (score) => {
  if (score >= 0.8) return "严重";
  if (score >= 0.5) return "警告";
  return "信息";
};

// 清除告警
const clearAlerts = () => {
  alerts.value = [];
};

// 控制拓扑图缩放
const zoomIn = () => {
  if (cy) {
    cy.zoom({
      level: cy.zoom() * 1.2,
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
    });
  }
};

const zoomOut = () => {
  if (cy) {
    cy.zoom({
      level: cy.zoom() * 0.8,
      renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
    });
  }
};

const resetView = () => {
  if (cy) {
    cy.fit();
    cy.center();
  }
};

// 页面初始化
onMounted(() => {
  loadNetworkTopology();
  initTrafficChart();

  // 启动两个定时器
  updateFlowTrendChart();
  setInterval(updateFlowTrendChart, 2000);

  updateGauges();
  setInterval(updateGauges, 2000);

  window.addEventListener("resize", () => {
    if (trafficChart) trafficChart.resize();
  });
});
</script>

<style scoped>
/* 全局样式 */
.cy-wrapper {
  position: relative;
  height: calc(100% - 50px);
}

.cy-container {
  width: 100%;
  height: 100%;
  box-shadow: inset 0 0 15px rgba(26, 32, 44, 0.5);
  background-color: #0a1929;
  position: relative;
  z-index: 1;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2;
  background-color: rgba(10, 25, 41, 0.85);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: #90cdf4;
  font-size: 14px;
}

.spinner {
  width: 36px;
  height: 36px;
  border: 4px solid #63b3ed;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 8px;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.dashboard-container {
  background-color: #0a1929;
  color: #e2e8f0;
  padding: 20px;
  min-height: 100vh;
  font-family: "Roboto", "Arial", sans-serif;
}

/* 标题区域 */
.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid rgba(99, 179, 237, 0.3);
}

.title-area h1 {
  font-size: 28px;
  font-weight: bold;
  color: #90cdf4;
  margin: 0;
  text-shadow: 0 0 10px rgba(144, 205, 244, 0.6);
}

.status-indicator {
  margin-top: 5px;
  font-size: 14px;
}

.status-active {
  color: #68d391;
  position: relative;
  padding-left: 18px;
}

.status-active::before {
  content: "";
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  width: 12px;
  height: 12px;
  background-color: #68d391;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

.stats-panel {
  display: flex;
  gap: 20px;
}

.stat-item {
  background-color: rgba(49, 130, 206, 0.2);
  border: 1px solid rgba(99, 179, 237, 0.3);
  border-radius: 8px;
  padding: 10px 20px;
  text-align: center;
  min-width: 100px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #90cdf4;
}

.stat-label {
  font-size: 12px;
  color: #a0aec0;
  margin-top: 4px;
}

/* 网格布局 */
.dashboard-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;
  height: calc(100vh - 150px);
}

/* 面板通用样式 */
.topology-panel,
.alert-panel,
.traffic-panel {
  background-color: rgba(26, 32, 44, 0.8);
  border: 1px solid rgba(99, 179, 237, 0.3);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 0 20px rgba(66, 153, 225, 0.2);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.panel-header {
  background-color: rgba(44, 82, 130, 0.5);
  padding: 10px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(99, 179, 237, 0.3);
}

.panel-header h2 {
  font-size: 18px;
  color: #90cdf4;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.panel-controls {
  display: flex;
  gap: 8px;
}

.control-btn {
  background-color: rgba(66, 153, 225, 0.3);
  border: 1px solid rgba(99, 179, 237, 0.5);
  border-radius: 4px;
  color: #90cdf4;
  font-size: 14px;
  padding: 4px 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.control-btn:hover {
  background-color: rgba(66, 153, 225, 0.5);
}

.icon {
  font-weight: bold;
}

/* 拓扑图容器 */
.cy-container {
  flex: 1;
  width: 100%;
  height: calc(100% - 50px);
  box-shadow: inset 0 0 15px rgba(26, 32, 44, 0.5);
  background-color: #0a1929;
  position: relative;
}

/* 告警面板 */
.side-panels {
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
}

.alert-panel {
  flex: 1;
}

.alert-content {
  padding: 15px;
  overflow-y: auto;
  max-height: 300px;
  /* ✅ 控制最大高度，防止页面撑大 */
  display: flex;
  flex-direction: column;
}

.traffic-content {
  padding: 15px;
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
}

/* 告警列表样式 */
.alert-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.alert-item {
  background-color: rgba(26, 32, 44, 0.5);
  border-left: 4px solid #90cdf4;
  border-radius: 6px;
  padding: 12px;
  animation: fadeIn 0.3s ease-in;
}

.alert-item.critical {
  border-left-color: #f56565;
  background-color: rgba(245, 101, 101, 0.1);
}

.alert-item.warning {
  border-left-color: #ed8936;
  background-color: rgba(237, 137, 54, 0.1);
}

.alert-item.info {
  border-left-color: #4299e1;
  background-color: rgba(66, 153, 225, 0.1);
}

.alert-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
}

.alert-timestamp {
  font-size: 12px;
  color: #a0aec0;
}

.alert-severity {
  font-size: 12px;
  font-weight: bold;
  padding: 2px 8px;
  border-radius: 10px;
  background-color: rgba(66, 153, 225, 0.2);
  color: #90cdf4;
}

.alert-item.critical .alert-severity {
  background-color: rgba(245, 101, 101, 0.2);
  color: #feb2b2;
}

.alert-item.warning .alert-severity {
  background-color: rgba(237, 137, 54, 0.2);
  color: #fbd38d;
}

.alert-details {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 14px;
}

.highlight {
  color: #fff;
  font-weight: bold;
}

.alert-confidence {
  display: flex;
  align-items: center;
  gap: 10px;
}

.confidence-bar {
  height: 8px;
  flex: 1;
  background-color: rgba(160, 174, 192, 0.2);
  border-radius: 4px;
  overflow: hidden;
}

.confidence-level {
  height: 100%;
  background-color: #4299e1;
  border-radius: 4px;
}

.alert-item.critical .confidence-level {
  background-color: #f56565;
}

.alert-item.warning .confidence-level {
  background-color: #ed8936;
}

.confidence-value {
  font-size: 12px;
  min-width: 40px;
  text-align: right;
}

.no-alerts {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 0;
  color: #a0aec0;
  gap: 15px;
}

.radar-animation {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background: radial-gradient(
    circle,
    rgba(56, 161, 105, 0.1) 0%,
    rgba(56, 161, 105, 0) 70%
  );
  position: relative;
}

.radar-animation::before {
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  width: 4px;
  height: 50px;
  background-color: rgba(56, 161, 105, 0.6);
  transform-origin: bottom center;
  animation: radar 3s linear infinite;
}

/* 流量分析样式 */
.traffic-chart {
  width: 100%;
  height: 200px;
  margin-bottom: 15px;
}

.traffic-stats {
  display: flex;
  justify-content: space-around;
  margin-top: 10px;
}

.traffic-stat {
  text-align: center;
}

.stat-circle {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin: 0 auto 10px;
  font-weight: bold;
  position: relative;
}

.stat-circle::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: 50%;
  border: 3px solid transparent;
  animation: rotate 4s linear infinite;
}

.stat-circle.incoming {
  background-color: rgba(144, 205, 244, 0.1);
}

.stat-circle.incoming::before {
  border-top-color: #90cdf4;
  border-right-color: #90cdf4;
}

.stat-circle.outgoing {
  background-color: rgba(154, 230, 180, 0.1);
}

.stat-circle.outgoing::before {
  border-top-color: #9ae6b4;
  border-left-color: #9ae6b4;
}

.stat-circle span {
  font-size: 20px;
  color: #e2e8f0;
}

.stat-circle small {
  font-size: 12px;
  color: #a0aec0;
}

/* 脉冲点样式 */
.pulse-dot {
  width: 12px;
  height: 12px;
  background-color: #63b3ed;
  border-radius: 50%;
  margin-right: 5px;
  display: inline-block;
}

.pulse-dot.active {
  background-color: #f56565;
  animation: pulse 2s infinite;
}

/* 动画 */
@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(245, 101, 101, 0.7);
  }

  70% {
    box-shadow: 0 0 0 10px rgba(245, 101, 101, 0);
  }

  100% {
    box-shadow: 0 0 0 0 rgba(245, 101, 101, 0);
  }
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }

  to {
    transform: rotate(360deg);
  }
}

@keyframes radar {
  from {
    transform: translate(-50%, 0) rotate(0deg);
  }

  to {
    transform: translate(-50%, 0) rotate(360deg);
  }
}

/* 响应式样式 */
@media screen and (max-width: 1200px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
  }

  .side-panels {
    grid-template-columns: 1fr 1fr;
    display: grid;
  }
}

@media screen and (max-width: 768px) {
  .dashboard-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }

  .stats-panel {
    width: 100%;
    justify-content: space-between;
  }

  .side-panels {
    grid-template-columns: 1fr;
  }

  .stat-item {
    min-width: unset;
    flex: 1;
  }
}

.traffic-panel {
  flex: 1;
}
</style>
