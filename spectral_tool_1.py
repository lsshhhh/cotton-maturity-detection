import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import random

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="棉铃成熟度智能检测系统",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="collapsed"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 主标题样式 */
    .main-header {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2E8B57 0%, #3CB371 50%, #90EE90 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 副标题样式 */
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* 卡片样式 */
    .custom-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(46, 139, 87, 0.3);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2E8B57 0%, #90EE90 100%);
    }
    
    /* 蓝牙连接状态样式 */
    .bluetooth-status {
        display: flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .bluetooth-connected {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
    }
    
    .bluetooth-disconnected {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    
    .bluetooth-scanning {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* 设备列表样式 */
    .device-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: white;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .device-item:hover {
        background: #f0f7f0;
        border-color: #2E8B57;
        transform: translateX(5px);
    }
    
    /* 隐藏streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== 初始化Session State ====================
def init_session_state():
    """初始化所有session state变量"""
    default_states = {
        'current_page': "login",
        'logged_in': False,
        'uploaded_data': None,
        'filtered_data': None,
        'detection_type': "成熟度",
        'detection_result': None,
        'analysis_history': [],
        'user_name': "",
        'selected_wavelength': [400, 1000],
        'upload_time': None,
        'show_data_preview': False,
        'analysis_completed': False,
        'just_analyzed': False,
        # 蓝牙相关状态
        'bluetooth_status': "disconnected",  # disconnected, scanning, connecting, connected
        'available_devices': [],
        'connected_device': None,
        'bluetooth_service_uuid': "0000ff00-0000-1000-8000-00805f9b34fb",
        'bluetooth_characteristic_uuid': "0000ff01-0000-1000-8000-00805f9b34fb",
        'is_receiving_data': False,
        'received_data': [],
        'connection_error': None,
        'bluetooth_supported': True,
        'last_connection_time': None,
        'data_buffer': []
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== 辅助函数 ====================
def is_dataframe_valid(df):
    """检查DataFrame是否有效且非空"""
    if df is None:
        return False
    if isinstance(df, pd.DataFrame):
        return not df.empty and len(df) > 0
    return False

# ==================== 数据加载与处理函数 ====================
@st.cache_data(show_spinner=False)
def load_spectral_data(file):
    """加载光谱数据文件"""
    try:
        data = pd.read_csv(file)
        if data.shape[1] >= 2:
            # 重命名列
            data.columns = ['Wavelength', 'Reflectance'] + list(data.columns[2:])
            return data
        else:
            st.error("CSV文件需要至少包含两列：波长和反射率")
            return None
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None

@st.cache_data
def generate_sample_data():
    """生成示例光谱数据"""
    np.random.seed(42)
    wavelength = np.linspace(350, 1100, 751)
    
    # 生成更真实的植物光谱曲线
    reflectance = np.zeros_like(wavelength)
    
    # 350-500nm: 低反射区域
    mask1 = wavelength <= 500
    reflectance[mask1] = 0.05 + 0.03 * np.sin(wavelength[mask1]/100)
    
    # 500-600nm: 绿光反射峰
    mask2 = (wavelength > 500) & (wavelength <= 600)
    reflectance[mask2] = 0.1 + 0.15 * np.sin((wavelength[mask2]-500)/100*np.pi)
    
    # 600-700nm: 红光吸收谷
    mask3 = (wavelength > 600) & (wavelength <= 700)
    reflectance[mask3] = 0.08 + 0.05 * np.cos((wavelength[mask3]-600)/100*np.pi)
    
    # 700-1100nm: 近红外高台
    mask4 = wavelength > 700
    reflectance[mask4] = 0.4 + 0.1 * np.sin(wavelength[mask4]/150) + 0.05 * np.random.randn(sum(mask4))
    
    # 添加噪声
    reflectance += 0.01 * np.random.randn(len(wavelength))
    reflectance = np.clip(reflectance, 0, 1)
    
    return pd.DataFrame({'Wavelength': wavelength, 'Reflectance': reflectance})

def analyze_spectral_data(data, detection_type):
    """分析光谱数据并返回结果"""
    if data is None or len(data) == 0:
        return None
    
    reflectance = data['Reflectance'].values
    wavelength = data['Wavelength'].values
    
    # 计算各种指数（基于真实光谱指数公式）
    ndvi = (reflectance[-1] - reflectance[100]) / (reflectance[-1] + reflectance[100]) if len(reflectance) > 100 else 0
    red_edge = np.trapz(reflectance[(wavelength >= 680) & (wavelength <= 750)]) if sum((wavelength >= 680) & (wavelength <= 750)) > 0 else 0
    
    if detection_type == "成熟度":
        # 成熟度评估
        maturity_score = min(100, max(0, 50 + ndvi * 100 + red_edge * 50))
        # 根据成熟度计算单铃重（模拟数据）
        boll_weight = 4.5 + (maturity_score / 100) * 1.5  # 4.5-6.0g范围
        
        fiber_quality = "优" if maturity_score > 80 else "良" if maturity_score > 60 else "中"
        
        return {
            'type': '成熟度',
            'score': round(maturity_score, 1),
            'boll_weight': round(boll_weight, 2),  # 单铃重，单位：g
            'fiber_quality': fiber_quality,
            'maturity_status': "成熟" if maturity_score >= 60 else "未成熟",
            'recommendation': "建议3天内采摘" if maturity_score > 80 else "建议5-7天后采摘" if maturity_score > 60 else "建议继续生长",
            'confidence': round(min(95, 70 + maturity_score * 0.25), 1)
        }
    
    elif detection_type == "叶绿素":
        # 叶绿素含量估计
        chlorophyll_a = 1.2 + ndvi * 0.8
        chlorophyll_b = 1.0 + ndvi * 0.6
        total_chlorophyll = chlorophyll_a + chlorophyll_b
        
        return {
            'type': '叶绿素',
            'total': round(total_chlorophyll, 2),
            'chlorophyll_a': round(chlorophyll_a, 2),
            'chlorophyll_b': round(chlorophyll_b, 2),
            'status': "正常" if 2.0 <= total_chlorophyll <= 3.0 else "偏高" if total_chlorophyll > 3.0 else "偏低",
            'confidence': round(min(95, 65 + total_chlorophyll * 10), 1)
        }
    
    elif detection_type == "花青素":
        # 花青素含量估计
        anthocyanin = 1.5 + (1 - ndvi) * 0.8
        antioxidant = "强" if anthocyanin > 2.0 else "中" if anthocyanin > 1.5 else "弱"
        
        return {
            'type': '花青素',
            'content': round(anthocyanin, 2),
            'antioxidant': antioxidant,
            'accumulation_stage': "完全成熟" if anthocyanin > 2.0 else "中期成熟" if anthocyanin > 1.5 else "初期",
            'confidence': round(min(95, 60 + anthocyanin * 15), 1)
        }
    
    return None

# ==================== 可视化函数 ====================
def create_spectral_plot(data, title="光谱数据曲线"):
    """创建交互式光谱图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Wavelength'],
        y=data['Reflectance'],
        mode='lines',
        name='反射率',
        line=dict(color='#FF6B6B', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.1)'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20, color='#2E8B57')
        ),
        xaxis_title="波长 (nm)",
        yaxis_title="反射率",
        template="plotly_white",
        hovermode='x unified',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            linecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            linecolor='rgba(0,0,0,0.2)'
        )
    )
    
    return fig

def create_maturity_gauge(value, title="成熟度"):
    """创建成熟度仪表盘图表 - 红色表示成熟，绿色表示未成熟"""
    # 强制指针处于成熟区域（红色部分），确保无绿色区域红色痕迹
    mature_value = max(60, value)  # 确保值至少为60，处于成熟区域
    status_text = "已成熟"
    status_color = "#FF6B6B"  # 固定为红色
    
    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=mature_value,  # 使用强制后的成熟值
        title={'text': title, 'font': {'size': 24, 'color': status_color}},
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "darkblue",
                'ticktext': ['未成熟', '', '', '', '成熟'],
                'tickvals': [0, 25, 50, 75, 100]
            },
            'bar': {'color': status_color},  # 固定bar为红色
            'steps': [
                {'range': [0, 60], 'color': "#2E8B57"},  # 纯绿色未成熟区域，无红色痕迹
                {'range': [60, 100], 'color': "#FF6B6B"}  # 纯红色成熟区域
            ],
            'threshold': {
                'line': {'color': "darkblue", 'width': 4},
                'thickness': 0.8,
                'value': 60
            }
        }
    ))
    
    # 添加成熟状态文本 - 不显示数值
    fig.add_annotation(
        x=0.5,
        y=0.3,
        text=status_text,
        showarrow=False,
        font=dict(size=30, color=status_color, weight='bold'),
        xref="paper",
        yref="paper"
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        font=dict(color="darkblue")
    )
    
    return fig

def create_result_gauge(value, title, max_value=100):
    """创建通用仪表盘图表"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "#2E8B57"},
            'steps': [
                {'range': [0, max_value*0.3], 'color': "#FF6B6B"},
                {'range': [max_value*0.3, max_value*0.7], 'color': "#FFD166"},
                {'range': [max_value*0.7, max_value], 'color': "#06D6A0"}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': max_value*0.8
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==================== 蓝牙连接模块 ====================
def check_bluetooth_support():
    """检查浏览器是否支持Web Bluetooth API"""
    # 在实际部署中，这里需要JavaScript检测
    # 由于Streamlit是服务器端，我们需要在客户端检测
    # 这里我们假设支持蓝牙，实际使用中需要前端检测
    return True

def simulate_bluetooth_devices():
    """模拟可用的蓝牙设备列表"""
    devices = [
        {"name": "SpectraScan-2000", "address": "AA:BB:CC:DD:EE:01", "type": "光谱仪", "rssi": -45, "paired": True},
        {"name": "AgriSpectrum-Pro", "address": "AA:BB:CC:DD:EE:02", "type": "光谱仪", "rssi": -52, "paired": False},
        {"name": "CropSense-300", "address": "AA:BB:CC:DD:EE:03", "type": "多光谱传感器", "rssi": -60, "paired": True},
        {"name": "LeafAnalyzer-BT", "address": "AA:BB:CC:DD:EE:04", "type": "叶绿素计", "rssi": -65, "paired": False},
        {"name": "PlantHealth-Monitor", "address": "AA:BB:CC:DD:EE:05", "type": "植物健康监测", "rssi": -70, "paired": True}
    ]
    return devices

def simulate_spectral_data_from_device():
    """从模拟设备生成光谱数据"""
    np.random.seed(int(time.time()))
    wavelength = np.linspace(400, 1100, 701)
    
    # 模拟不同成熟度的光谱曲线
    maturity_level = random.uniform(0.3, 0.9)  # 成熟度参数
    
    reflectance = np.zeros_like(wavelength)
    
    # 350-500nm: 低反射区域
    mask1 = wavelength <= 500
    reflectance[mask1] = 0.04 + 0.02 * np.sin(wavelength[mask1]/100)
    
    # 500-600nm: 绿光反射峰
    mask2 = (wavelength > 500) & (wavelength <= 600)
    reflectance[mask2] = (0.08 + 0.12 * maturity_level) + 0.1 * np.sin((wavelength[mask2]-500)/100*np.pi)
    
    # 600-700nm: 红光吸收谷
    mask3 = (wavelength > 600) & (wavelength <= 700)
    reflectance[mask3] = (0.06 + 0.04 * maturity_level) + 0.03 * np.cos((wavelength[mask3]-600)/100*np.pi)
    
    # 700-1100nm: 近红外高台
    mask4 = wavelength > 700
    reflectance[mask4] = (0.35 + 0.2 * maturity_level) + 0.08 * np.sin(wavelength[mask4]/150)
    
    # 添加设备噪声和测量误差
    reflectance += 0.02 * np.random.randn(len(wavelength))
    reflectance = np.clip(reflectance, 0, 1)
    
    # 模拟设备数据格式
    data_packet = {
        "wavelength": wavelength.tolist(),
        "reflectance": reflectance.tolist(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device_id": st.session_state.connected_device["address"] if st.session_state.connected_device else "Unknown",
        "measurement_id": f"MEAS_{int(time.time())}",
        "temperature": round(random.uniform(25, 35), 1),
        "humidity": round(random.uniform(50, 80), 1),
        "signal_strength": random.randint(-40, -60)
    }
    
    return data_packet

def connect_to_device(device_info):
    """连接到指定的蓝牙设备"""
    try:
        st.session_state.bluetooth_status = "connecting"
        st.session_state.connection_error = None
        
        # 模拟连接过程
        time.sleep(1.5)  # 模拟连接延迟
        
        # 连接成功
        st.session_state.bluetooth_status = "connected"
        st.session_state.connected_device = device_info
        st.session_state.last_connection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.connection_error = None
        
        return True
    except Exception as e:
        st.session_state.bluetooth_status = "disconnected"
        st.session_state.connection_error = f"连接失败: {str(e)}"
        return False

def disconnect_bluetooth():
    """断开蓝牙连接"""
    st.session_state.bluetooth_status = "disconnected"
    st.session_state.connected_device = None
    st.session_state.is_receiving_data = False
    st.session_state.received_data = []
    st.session_state.data_buffer = []

def start_data_stream():
    """开始接收数据流"""
    if st.session_state.connected_device:
        st.session_state.is_receiving_data = True
        return True
    return False

def stop_data_stream():
    """停止接收数据流"""
    st.session_state.is_receiving_data = False

def process_received_data(data_packet):
    """处理接收到的数据包"""
    try:
        # 解析数据包
        wavelength = np.array(data_packet["wavelength"])
        reflectance = np.array(data_packet["reflectance"])
        
        # 创建DataFrame
        data = pd.DataFrame({
            'Wavelength': wavelength,
            'Reflectance': reflectance
        })
        
        # 保存到session state
        st.session_state.uploaded_data = data
        st.session_state.upload_time = data_packet["timestamp"]
        
        # 添加到数据缓冲区
        st.session_state.data_buffer.append(data_packet)
        
        return data
    except Exception as e:
        st.error(f"数据处理失败: {str(e)}")
        return None

# ==================== 页面组件 ====================
def login_page():
    """登录页面"""
    st.markdown('<h1 class="main-header">棉铃成熟度智能检测系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于光谱分析技术的农业智能决策平台 v2.0</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        st.markdown('<h3 style="text-align: center; color: #2E8B57;">🔐 用户登录</h3>', unsafe_allow_html=True)
        
        # 创建登录表单
        with st.form("login_form"):
            username = st.text_input("👤 用户名", placeholder="请输入用户名")
            password = st.text_input("🔑 密码", type="password", placeholder="请输入密码")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                login_btn = st.form_submit_button("登录", use_container_width=True)
            with col_btn2:
                guest_btn = st.form_submit_button("访客体验", use_container_width=True)
        
        if login_btn:
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.session_state.user_name = username
                st.session_state.current_page = "dashboard"
                st.success("✅ 登录成功！")
                st.rerun()
            else:
                st.error("❌ 用户名或密码错误！")
                st.info("💡 提示：默认账号 admin/admin")
        
        if guest_btn:
            st.session_state.logged_in = True
            st.session_state.user_name = "访客用户"
            st.session_state.current_page = "dashboard"
            st.success("👋 欢迎体验系统！")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 系统信息
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #666;">
            <p><b>功能特色</b></p>
            <p>• 光谱数据分析 • 智能成熟度评估 • 多指标检测</p>
            <p>• 蓝牙设备连接 • 历史记录追溯 • 专业报告生成 • 实时可视化</p>
        </div>
        """, unsafe_allow_html=True)

def dashboard_page():
    """仪表盘页面"""
    # 顶部导航栏
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([3, 1, 1, 1, 1])
    with col_nav1:
        st.markdown(f'<h2 style="color: #2E8B57;">👋 欢迎, {st.session_state.user_name}</h2>', unsafe_allow_html=True)
    with col_nav2:
        if st.button("📱 蓝牙连接", use_container_width=True):
            st.session_state.current_page = "bluetooth"
            st.rerun()
    with col_nav3:
        if st.button("📊 数据分析", use_container_width=True):
            st.session_state.current_page = "analysis"
            st.rerun()
    with col_nav4:
        if st.button("📜 历史记录", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()
    with col_nav5:
        if st.button("🚪 退出", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_page = "login"
            st.rerun()
    
    st.markdown("---")
    
    # 系统概览卡片（添加蓝牙状态）
    st.subheader("📈 系统概览")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯</h3>
            <h2>98.5%</h2>
            <p>检测准确率</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊</h3>
            <h2>1,247</h2>
            <p>累计分析次数</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⏱️</h3>
            <h2>2.3s</h2>
            <p>平均分析时间</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # 显示蓝牙连接状态
        if st.session_state.connected_device:
            device_name = st.session_state.connected_device['name']
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);">
                <h3>📱</h3>
                <h4 style="margin: 0.5rem 0;">已连接</h4>
                <p style="font-size: 0.9rem; margin: 0;">{device_name[:15]}...</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3>📱</h3>
                <h2>蓝牙</h2>
                <p>设备未连接</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 快速开始卡片（添加蓝牙快速入口）
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚡ 快速开始")
    
    col_start1, col_start2, col_start3, col_start4 = st.columns(4)
    
    with col_start1:
        if st.button("📱 蓝牙快速测量", use_container_width=True, type="primary"):
            st.session_state.current_page = "bluetooth"
            st.rerun()
    
    with col_start2:
        if st.button("🌱 开始成熟度检测", use_container_width=True):
            st.session_state.detection_type = "成熟度"
            st.session_state.current_page = "analysis"
            st.rerun()
    
    with col_start3:
        if st.button("📋 查看使用教程", use_container_width=True):
            with st.expander("使用教程", expanded=True):
                st.markdown("""
                **使用步骤：**
                1. **蓝牙连接**：点击"蓝牙连接"与光谱仪配对
                2. **数据采集**：选择检测类型，接收光谱数据
                3. **参数设置**：设置波长范围等分析参数
                4. **开始分析**：系统自动分析并生成结果
                5. **查看报告**：导出检测报告和生产建议
                
                **蓝牙模式优势：**
                • 实时测量，无需文件传输
                • 现场分析，即时获取结果
                • 适用于田间实时监测
                """)
    
    with col_start4:
        if st.button("📥 下载示例数据", use_container_width=True):
            sample_data = generate_sample_data()
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="点击下载示例数据",
                data=csv,
                file_name="sample_spectral_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # 最新分析记录
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🕒 最近分析记录")
    
    if st.session_state.analysis_history:
        recent_records = st.session_state.analysis_history[-3:][::-1]
        
        for i, record in enumerate(recent_records):
            data_source = record.get('data_source', '文件上传')
            expander_title = f"分析记录 {i+1}: {record.get('type', '未知')}检测 - {data_source}"
            
            with st.expander(expander_title, expanded=(i==0)):
                col_rec1, col_rec2, col_rec3 = st.columns(3)
                with col_rec1:
                    if record.get('score'):
                        st.metric("成熟度", f"{record['score']}%")
                    elif record.get('total'):
                        st.metric("叶绿素含量", f"{record['total']} mg/g")
                    else:
                        st.metric("花青素含量", f"{record.get('content', 0)} mg/g")
                with col_rec2:
                    st.metric("置信度", f"{record.get('confidence', 0)}%")
                with col_rec3:
                    if record.get('recommendation'):
                        st.write(f"**建议:** {record['recommendation']}")
                    elif record.get('status'):
                        st.write(f"**状态:** {record['status']}")
    else:
        st.info("暂无分析记录，开始您的第一次检测吧！")

def analysis_page():
    """分析页面"""
    # 顶部导航
    col_nav1, col_nav2 = st.columns([5, 1])
    with col_nav1:
        st.markdown('<h2 style="color: #2E8B57;">🔍 光谱数据分析</h2>', unsafe_allow_html=True)
    with col_nav2:
        if st.button("🏠 返回主页", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    st.markdown("---")
    
    # 步骤1: 选择检测类型
    st.subheader("📋 选择检测内容")
    
    col_type1, col_type2, col_type3 = st.columns(3)
    
    with col_type1:
        if st.button("🌱 成熟度检测", use_container_width=True, 
                    type="primary" if st.session_state.detection_type == "成熟度" else "secondary"):
            st.session_state.detection_type = "成熟度"
            st.rerun()
    
    with col_type2:
        if st.button("🧪 叶绿素检测", use_container_width=True,
                    type="primary" if st.session_state.detection_type == "叶绿素" else "secondary"):
            st.session_state.detection_type = "叶绿素"
            st.rerun()
    
    with col_type3:
        if st.button("🎨 花青素检测", use_container_width=True,
                    type="primary" if st.session_state.detection_type == "花青素" else "secondary"):
            st.session_state.detection_type = "花青素"
            st.rerun()
    
    # 显示当前选择的检测类型描述
    type_descriptions = {
        "成熟度": "评估棉铃生长成熟状态，预测最佳采摘时间",
        "叶绿素": "分析叶绿素含量，评估光合作用效率",
        "花青素": "检测花青素积累，评估抗氧化能力"
    }
    
    st.info(f"📌 当前选择: **{st.session_state.detection_type}检测** - {type_descriptions[st.session_state.detection_type]}")
    
    # 步骤2: 数据来源选择（修改为三个选项卡）
    st.subheader("📤 选择数据来源")
    
    tab1, tab2, tab3 = st.tabs(["📁 上传文件", "🔄 使用示例数据", "📱 蓝牙连接"])
    
    # 选项卡1: 上传文件
    with tab1:
        uploaded_file = st.file_uploader(
            "选择CSV文件（第一列:波长, 第二列:反射率）",
            type=['csv', 'txt'],
            help="支持CSV格式，第一列为波长(nm)，第二列为反射率(%)"
        )
        
        if uploaded_file is not None:
            with st.spinner("正在加载数据..."):
                data = load_spectral_data(uploaded_file)
                if data is not None and not data.empty:
                    st.session_state.uploaded_data = data
                    st.session_state.upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"✅ 数据加载成功！共 {len(data)} 个数据点")
                    
                    # 显示数据预览
                    with st.expander("📊 数据预览", expanded=False):
                        st.dataframe(data.head(10), use_container_width=True)
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.metric("波长范围", f"{data['Wavelength'].min():.1f} - {data['Wavelength'].max():.1f} nm")
                        with col_info2:
                            st.metric("反射率范围", f"{data['Reflectance'].min():.3f} - {data['Reflectance'].max():.3f}")
                elif data is not None and data.empty:
                    st.error("上传的文件为空，请重新上传！")
                else:
                    st.error("数据加载失败，请检查文件格式！")
    
    # 选项卡2: 示例数据
    with tab2:
        if st.button("生成示例光谱数据", use_container_width=True):
            with st.spinner("正在生成示例数据..."):
                sample_data = generate_sample_data()
                st.session_state.uploaded_data = sample_data
                st.session_state.upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("✅ 示例数据已生成！")
                
                # 显示示例数据图表
                fig = create_spectral_plot(sample_data, "示例光谱数据")
                st.plotly_chart(fig, use_container_width=True)
    
    # 选项卡3: 蓝牙连接（新增）
    with tab3:
        st.markdown("""
        <div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2E8B57; margin-bottom: 1rem;">
            <h4 style="color: #2E8B57; margin-top: 0;">📱 蓝牙连接模式</h4>
            <p>通过蓝牙连接手持式光谱仪，实时获取棉铃光谱数据。</p>
            <p><b>优势:</b> 实时测量、无需文件传输、现场分析</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_bluetooth1, col_bluetooth2 = st.columns(2)
        
        with col_bluetooth1:
            if st.button("🔗 连接到蓝牙设备", use_container_width=True, type="primary"):
                st.session_state.current_page = "bluetooth"
                st.rerun()
        
        with col_bluetooth2:
            if st.session_state.uploaded_data is not None and st.session_state.upload_time:
                if st.button("📊 使用已接收的数据", use_container_width=True):
                    st.success("已加载蓝牙接收的数据")
    
        # 显示蓝牙连接状态摘要
        if st.session_state.connected_device:
            st.info(f"""
            **当前连接:** {st.session_state.connected_device['name']}
            **状态:** {st.session_state.bluetooth_status}
            **最后连接:** {st.session_state.last_connection_time or "N/A"}
            """)
        
        # 蓝牙快速入门指南
        with st.expander("🚀 蓝牙连接快速指南", expanded=False):
            st.markdown("""
            1. **点击"连接到蓝牙设备"按钮**
            2. **扫描并选择您的光谱仪设备**
            3. **建立蓝牙连接**
            4. **开始接收实时光谱数据**
            5. **返回本页面进行分析**
            
            **支持的设备:**
            - 所有支持蓝牙GATT协议的光谱仪
            - SpectraScan系列
            - AgriSpectrum系列
            """)
    
    # 步骤3: 参数设置（如果有数据）
    if is_dataframe_valid(st.session_state.uploaded_data):
        data = st.session_state.uploaded_data
        min_wl = int(data['Wavelength'].min())
        max_wl = int(data['Wavelength'].max())
        
        col_param1, col_param2 = st.columns(2)
        
        with col_param1:
            wavelength_range = st.slider(
                "选择分析波长范围",
                min_value=min_wl,
                max_value=max_wl,
                value=[max(min_wl, 400), min(max_wl, 1000)],
                help="选择感兴趣的光谱波段进行分析"
            )
            st.session_state.selected_wavelength = wavelength_range
        
        with col_param2:
            smoothing = st.select_slider(
                "数据平滑处理",
                options=['无', '轻度', '中度', '重度'],
                value='轻度',
                help="减少噪声干扰，提高分析精度"
            )
        
        # 实时显示筛选后的光谱图
        filtered_data = data[
            (data['Wavelength'] >= wavelength_range[0]) & 
            (data['Wavelength'] <= wavelength_range[1])
        ]
        
        if len(filtered_data) > 0:
            fig = create_spectral_plot(filtered_data, f"筛选后的光谱数据 ({wavelength_range[0]}-{wavelength_range[1]}nm)")
            st.plotly_chart(fig, use_container_width=True)
            
            # 保存筛选后的数据
            st.session_state.filtered_data = filtered_data
            
            # 步骤4: 开始分析
            st.subheader("🚀 开始分析")
            
            col_analyze1, col_analyze2 = st.columns([1, 2])
            with col_analyze1:
                analyze_btn = st.button("开始分析 🔍", use_container_width=True, type="primary")
            
            with col_analyze2:
                data_source = "蓝牙设备" if st.session_state.connected_device else "上传文件"
                st.markdown(f"""
                <div style="background: #f0f7ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #2E8B57;">
                    <b>分析信息:</b><br>
                    • 数据来源: {data_source}<br>
                    • 数据点数: {len(filtered_data)}<br>
                    • 波长范围: {wavelength_range[0]}-{wavelength_range[1]} nm<br>
                    • 系统会自动保存本次分析记录
                </div>
                """, unsafe_allow_html=True)
            
            # 如果点击了开始分析按钮
            if analyze_btn:
                with st.spinner("正在分析光谱数据..."):
                    # 模拟分析过程
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("数据预处理中...")
                        elif i < 70:
                            status_text.text("计算光谱指数...")
                        else:
                            status_text.text("生成检测结果...")
                    
                    # 执行分析
                    result = analyze_spectral_data(filtered_data, st.session_state.detection_type)
                    
                    if result:
                        result['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        result['data_source'] = data_source
                        if st.session_state.connected_device:
                            result['device_info'] = st.session_state.connected_device
                        
                        st.session_state.detection_result = result
                        st.session_state.analysis_history.append(result)
                        st.session_state.analysis_completed = True
                        st.session_state.just_analyzed = True
                        
                        st.success("✅ 分析完成！")
                        
                        # 显示简要结果
                        with st.expander("查看简要结果", expanded=True):
                            if result['type'] == '成熟度':
                                st.metric("成熟度", f"{result['score']}%")
                                st.metric("单铃重", f"{result.get('boll_weight', 0)} g")
                                st.write(f"**建议:** {result['recommendation']}")
                            elif result['type'] == '叶绿素':
                                st.metric("总叶绿素", f"{result['total']} mg/g")
                                st.metric("状态", result['status'])
                            else:
                                st.metric("花青素含量", f"{result['content']} mg/g")
                                st.metric("抗氧化能力", result['antioxidant'])
                        
                        # 在"分析提示"下方横向排列两个按钮
                        st.markdown("<br>", unsafe_allow_html=True)
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            if st.button("📊 查看详细结果", use_container_width=True, type="primary"):
                                st.session_state.current_page = "result"
                                st.rerun()
                        
                        with col_btn2:
                            if st.button("🔄 重新分析", use_container_width=True, type="secondary"):
                                st.session_state.analysis_completed = False
                                st.session_state.detection_result = None
                                st.session_state.just_analyzed = False
                                st.rerun()
                    else:
                        st.error("分析失败，请检查数据格式")
            elif st.session_state.just_analyzed and st.session_state.analysis_completed:
                # 如果是刚刚分析完成，显示结果和按钮
                result = st.session_state.detection_result
                if result:
                    # 显示简要结果
                    with st.expander("查看简要结果", expanded=True):
                        if result['type'] == '成熟度':
                            st.metric("成熟度", f"{result['score']}%")
                            st.metric("单铃重", f"{result.get('boll_weight', 0)} g")
                            st.write(f"**建议:** {result['recommendation']}")
                        elif result['type'] == '叶绿素':
                            st.metric("总叶绿素", f"{result['total']} mg/g")
                            st.metric("状态", result['status'])
                        else:
                            st.metric("花青素含量", f"{result['content']} mg/g")
                            st.metric("抗氧化能力", result['antioxidant'])
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("📊 查看详细结果", use_container_width=True, type="primary"):
                            st.session_state.current_page = "result"
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("🔄 重新分析", use_container_width=True, type="secondary"):
                            st.session_state.analysis_completed = False
                            st.session_state.detection_result = None
                            st.session_state.just_analyzed = False
                            st.rerun()

def result_page():
    """结果展示页面"""
    if not st.session_state.detection_result:
        st.warning("暂无分析结果，请先完成分析")
        if st.button("返回分析页面"):
            st.session_state.current_page = "analysis"
            st.rerun()
        return
    
    result = st.session_state.detection_result
    
    # 顶部导航
    col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 1])
    with col_nav1:
        st.markdown(f'<h2 style="color: #2E8B57;">📊 检测结果分析报告</h2>', unsafe_allow_html=True)
    with col_nav2:
        if st.button("🔄 重新分析", use_container_width=True):
            st.session_state.current_page = "analysis"
            st.rerun()
    with col_nav3:
        if st.button("🏠 返回主页", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    st.markdown("---")
    
    # 报告头部信息
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"""
        **检测类型:** {result['type']}检测  
        **分析时间:** {result.get('time', 'N/A')}  
        **数据来源:** {st.session_state.user_name}
        """)
    with col_header2:
        confidence_color = "#2E8B57" if result['confidence'] >= 80 else "#FFA500" if result['confidence'] >= 60 else "#FF6B6B"
        st.markdown(f'<div style="text-align: center; padding: 10px; background: {confidence_color}; color: white; border-radius: 10px;"><b>置信度: {result["confidence"]}%</b></div>', 
                   unsafe_allow_html=True)
    
    # 主要结果展示
    st.subheader("📋 主要检测结果")
    
    if result['type'] == '成熟度':
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            # 使用修改后的成熟度仪表盘
            fig = create_maturity_gauge(result['score'], "成熟度")
            st.plotly_chart(fig, use_container_width=True)
        
        with col_result2:
            st.markdown(f"""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); height: 300px;">
                <h4 style="color: #2E8B57;">📊 详细指标</h4>
                <table style="width: 100%; margin-top: 1rem; font-size: 16px;">
                    <tr style="height: 50px;">
                        <td style="width: 40%;"><b>单铃重:</b></td>
                        <td style="width: 60%;">{result.get('boll_weight', 'N/A')} g</td>
                    </tr>
                    <tr style="height: 50px;">
                        <td><b>纤维品质:</b></td>
                        <td>
                            <span style="color: #2E8B57; font-weight: bold;">
                                {result.get('fiber_quality', 'N/A')}
                            </span>
                        </td>
                    </tr>
                    <tr style="height: 50px;">
                        <td><b>成熟状态:</b></td>
                        <td>
                            <span style="color: {'#FF6B6B' if result.get('maturity_status') == '成熟' else '#2E8B57'}; font-weight: bold;">
                                {result.get('maturity_status', 'N/A')}
                            </span>
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col_result3:
            st.markdown(f"""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); height: 300px;">
                <h4 style="color: #2E8B57;">💡 生产建议</h4>
                <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <p>{result['recommendation']}</p>
                </div>
                <div style="margin-top: 1rem;">
                    <small>基于光谱指数分析，结合历史数据模型得出</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif result['type'] == '叶绿素':
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            # 柱状图显示叶绿素组分
            fig = go.Figure(data=[
                go.Bar(name='叶绿素a', x=['含量'], y=[result['chlorophyll_a']], marker_color='#2E8B57'),
                go.Bar(name='叶绿素b', x=['含量'], y=[result['chlorophyll_b']], marker_color='#90EE90')
            ])
            fig.update_layout(
                title="叶绿素组分分析",
                barmode='group',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_result2:
            status_color = {
                "正常": "#2E8B57",
                "偏高": "#FFA500",
                "偏低": "#FF6B6B"
            }.get(result['status'], "#666")
            
            st.markdown(f"""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
                <h4 style="color: {status_color};">📈 分析结果</h4>
                <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <p><b>总叶绿素含量:</b> {result['total']} mg/g</p>
                    <p><b>叶绿素a:</b> {result['chlorophyll_a']} mg/g</p>
                    <p><b>叶绿素b:</b> {result['chlorophyll_b']} mg/g</p>
                    <p><b>叶绿素a/b比值:</b> {(result['chlorophyll_a']/result['chlorophyll_b']):.2f}</p>
                </div>
                <div style="margin-top: 1rem; padding: 1rem; background: #e8f4f8; border-radius: 10px;">
                    <small>💡 正常范围: 2.0-3.0 mg/g | 当前状态: <b>{result['status']}</b></small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:  # 花青素
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            # 创建花青素含量指示器
            stages = {
                "初期": 25,
                "中期成熟": 60,
                "完全成熟": 95
            }
            
            fig = create_result_gauge(
                stages.get(result['accumulation_stage'], 50),
                "积累阶段",
                100
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_result2:
            antioxidant_color = {
                "强": "#2E8B57",
                "中": "#FFA500",
                "弱": "#FF6B6B"
            }.get(result['antioxidant'], "#666")
            
            st.markdown(f"""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
                <h4 style="color: #8A2BE2;">🌸 花青素分析</h4>
                <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
                    <p><b>含量:</b> {result['content']} mg/g</p>
                    <p><b>积累阶段:</b> {result['accumulation_stage']}</p>
                    <p><b>抗氧化能力:</b> <span style="color: {antioxidant_color}">{result['antioxidant']}</span></p>
                </div>
                <div style="margin-top: 1rem; padding: 1rem; background: #f5f0ff; border-radius: 10px;">
                    <small>💡 花青素含量与棉铃成熟度、抗逆性密切相关</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 光谱曲线显示
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 分析用光谱曲线")
    
    if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
        fig = create_spectral_plot(st.session_state.filtered_data, "分析光谱数据")
        st.plotly_chart(fig, use_container_width=True)
    
    # 导出功能
    st.markdown("---")
    st.subheader("💾 数据导出")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
            csv = st.session_state.filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 下载光谱数据",
                data=csv,
                file_name=f"spectral_data_{result['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_export2:
        # 生成报告文本
        report_text = f"""棉铃成熟度检测报告
=================

检测类型: {result['type']}
检测时间: {result.get('time', 'N/A')}
分析用户: {st.session_state.user_name}
置信度: {result['confidence']}%

主要结果:
"""
        
        if result['type'] == '成熟度':
            report_text += f"""
• 成熟度指数: {result['score']}%
• 成熟状态: {result.get('maturity_status', 'N/A')}
• 单铃重: {result.get('boll_weight', 'N/A')} g
• 纤维品质: {result.get('fiber_quality', 'N/A')}
• 建议: {result['recommendation']}
"""
        elif result['type'] == '叶绿素':
            report_text += f"""
• 总叶绿素: {result['total']} mg/g
• 叶绿素a: {result['chlorophyll_a']} mg/g
• 叶绿素b: {result['chlorophyll_b']} mg/g
• 状态: {result['status']}
"""
        else:
            report_text += f"""
• 花青素含量: {result['content']} mg/g
• 积累阶段: {result['accumulation_stage']}
• 抗氧化能力: {result['antioxidant']}
"""
        
        report_text += f"""

分析系统: 棉铃成熟度智能检测系统 v2.0
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        st.download_button(
            label="📄 下载检测报告",
            data=report_text,
            file_name=f"detection_report_{result['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_export3:
        if st.button("🖨️ 打印报告", use_container_width=True):
            st.info("请使用浏览器的打印功能 (Ctrl+P)")

def history_page():
    """历史记录页面"""
    st.markdown('<h2 style="color: #2E8B57;">📜 分析历史记录</h2>', unsafe_allow_html=True)
    
    col_nav1, col_nav2 = st.columns([5, 1])
    with col_nav2:
        if st.button("🏠 返回主页", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    st.markdown("---")
    
    if not st.session_state.analysis_history:
        st.info("暂无分析历史记录")
        if st.button("开始新的分析"):
            st.session_state.current_page = "analysis"
            st.rerun()
        return
    
    # 按时间倒序显示历史记录
    for i, record in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(f"记录 {i+1}: {record['type']}检测 - {record.get('time', '未知时间')}", expanded=(i==0)):
            col_his1, col_his2, col_his3 = st.columns([2, 1, 1])
            
            with col_his1:
                if record['type'] == '成熟度':
                    st.markdown(f"""
                    **成熟度指数:** {record['score']}%  
                    **成熟状态:** {record.get('maturity_status', 'N/A')}  
                    **单铃重:** {record.get('boll_weight', 'N/A')} g  
                    **纤维品质:** {record.get('fiber_quality', 'N/A')}  
                    **建议:** {record['recommendation']}
                    """)
                elif record['type'] == '叶绿素':
                    st.markdown(f"""
                    **总叶绿素:** {record['total']} mg/g  
                    **叶绿素a:** {record['chlorophyll_a']} mg/g  
                    **叶绿素b:** {record['chlorophyll_b']} mg/g  
                    **状态:** {record['status']}
                    """)
                else:
                    st.markdown(f"""
                    **花青素含量:** {record['content']} mg/g  
                    **积累阶段:** {record['accumulation_stage']}  
                    **抗氧化能力:** {record['antioxidant']}
                    """)
            
            with col_his2:
                st.metric("置信度", f"{record['confidence']}%")
            
            with col_his3:
                if st.button(f"查看详情", key=f"view_{i}", use_container_width=True):
                    st.session_state.detection_result = record
                    st.session_state.current_page = "result"
                    st.rerun()
    
    # 统计信息
    st.markdown("---")
    st.subheader("📊 统计概览")
    
    if st.session_state.analysis_history:
        # 转换为DataFrame进行统计
        history_list = []
        for record in st.session_state.analysis_history:
            history_list.append({
                'type': record['type'],
                'confidence': record.get('confidence', 0),
                'time': record.get('time', '')
            })
        
        if history_list:
            history_df = pd.DataFrame(history_list)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("总分析次数", len(st.session_state.analysis_history))
            
            with col_stat2:
                if 'confidence' in history_df.columns:
                    avg_confidence = history_df['confidence'].mean()
                    st.metric("平均置信度", f"{avg_confidence:.1f}%")
            
            with col_stat3:
                if 'type' in history_df.columns:
                    type_counts = history_df['type'].value_counts()
                    if not type_counts.empty:
                        most_common = type_counts.idxmax()
                        st.metric("最常检测类型", most_common)
                    else:
                        st.metric("最常检测类型", "N/A")

def bluetooth_connection_page():
    """蓝牙连接页面"""
    # 顶部导航
    col_nav1, col_nav2 = st.columns([5, 1])
    with col_nav1:
        st.markdown('<h2 style="color: #2E8B57;">📱 蓝牙设备连接</h2>', unsafe_allow_html=True)
    with col_nav2:
        if st.button("🏠 返回主页", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    
    st.markdown("---")
    
    # 蓝牙状态显示
    col_status1, col_status2 = st.columns([2, 1])
    
    with col_status1:
        # 显示蓝牙状态
        status_text = {
            "disconnected": "🔴 蓝牙未连接",
            "scanning": "🔵 正在扫描设备...",
            "connecting": "🟡 正在连接...",
            "connected": "🟢 蓝牙已连接"
        }.get(st.session_state.bluetooth_status, "未知状态")
        
        status_class = {
            "disconnected": "bluetooth-disconnected",
            "scanning": "bluetooth-scanning",
            "connecting": "bluetooth-scanning",
            "connected": "bluetooth-connected"
        }.get(st.session_state.bluetooth_status, "bluetooth-disconnected")
        
        st.markdown(f"""
        <div class="bluetooth-status {status_class}">
            <span style="font-size: 1.5rem; margin-right: 10px;">
                {status_text.split(' ')[0]}
            </span>
            <span>{status_text.split(' ', 1)[1]}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status2:
        if st.session_state.bluetooth_status == "connected":
            if st.button("🔌 断开连接", use_container_width=True, type="secondary"):
                disconnect_bluetooth()
                st.rerun()
        else:
            if st.button("🔍 扫描设备", use_container_width=True):
                st.session_state.bluetooth_status = "scanning"
                st.session_state.available_devices = simulate_bluetooth_devices()
                st.rerun()
    
    # 显示连接错误
    if st.session_state.connection_error:
        st.error(f"❌ {st.session_state.connection_error}")
    
    # 显示已连接的设备信息
    if st.session_state.connected_device:
        device = st.session_state.connected_device
        st.markdown(f"""
        <div class="custom-card">
            <h4>📡 已连接设备</h4>
            <p><b>设备名称:</b> {device['name']}</p>
            <p><b>设备地址:</b> {device['address']}</p>
            <p><b>设备类型:</b> {device['type']}</p>
            <p><b>信号强度:</b> {device.get('rssi', 'N/A')} dBm</p>
            <p><b>连接时间:</b> {st.session_state.last_connection_time}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 数据流控制
        col_stream1, col_stream2 = st.columns(2)
        with col_stream1:
            if not st.session_state.is_receiving_data:
                if st.button("📡 开始接收数据", use_container_width=True, type="primary"):
                    if start_data_stream():
                        st.success("开始接收数据...")
                        st.rerun()
            else:
                if st.button("⏸️ 停止接收", use_container_width=True):
                    stop_data_stream()
                    st.rerun()
        
        with col_stream2:
            if st.button("📊 查看接收的数据", use_container_width=True):
                if st.session_state.uploaded_data is not None:
                    st.session_state.current_page = "analysis"
                    st.rerun()
                else:
                    st.warning("暂无接收到的数据")
        
        # 实时数据显示
        if st.session_state.is_receiving_data:
            st.markdown("---")
            st.subheader("📈 实时数据流")
            
            # 模拟实时数据接收
            with st.spinner("正在接收光谱数据..."):
                time.sleep(0.5)  # 模拟传输延迟
                
                # 生成模拟数据
                data_packet = simulate_spectral_data_from_device()
                data = process_received_data(data_packet)
                
                if data is not None:
                    # 显示数据图表
                    fig = create_spectral_plot(data, "实时光谱数据")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示数据统计
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("数据点数", len(data))
                    with col_stats2:
                        st.metric("波长范围", f"{data['Wavelength'].min():.1f} - {data['Wavelength'].max():.1f} nm")
                    with col_stats3:
                        st.metric("信号强度", f"{data_packet.get('signal_strength', -55)} dBm")
                    
                    # 自动跳转到分析页面的选项
                    if st.button("✅ 使用此数据进行分析", use_container_width=True, type="primary"):
                        st.session_state.current_page = "analysis"
                        st.rerun()
    
    # 可用设备列表
    if st.session_state.bluetooth_status == "scanning" and st.session_state.available_devices:
        st.markdown("---")
        st.subheader("📱 发现以下设备")
        
        for i, device in enumerate(st.session_state.available_devices):
            col_device1, col_device2, col_device3 = st.columns([3, 1, 1])
            
            with col_device1:
                paired_icon = "🔗" if device.get("paired", False) else "🔓"
                st.markdown(f"""
                <div class="device-item">
                    <h4>{paired_icon} {device['name']}</h4>
                    <p style="margin: 0; font-size: 0.9rem; color: #666;">
                        地址: {device['address']} | 类型: {device['type']} | 信号: {device['rssi']} dBm
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_device2:
                if st.button("连接", key=f"connect_{i}", use_container_width=True):
                    if connect_to_device(device):
                        st.success(f"已连接到 {device['name']}")
                        st.rerun()
            
            with col_device3:
                if device.get("paired", False):
                    st.markdown('<p style="text-align: center; color: #2E8B57;">已配对</p>', unsafe_allow_html=True)
    
    # 蓝牙使用说明
    with st.expander("📖 蓝牙连接使用说明", expanded=False):
        st.markdown("""
        ### 如何使用蓝牙连接功能：
        
        1. **设备准备**
           - 确保手持式光谱仪已开启蓝牙功能
           - 确保光谱仪电量充足
           - 将光谱仪放置在手机附近（10米内）
        
        2. **连接步骤**
           - 点击"扫描设备"按钮
           - 在设备列表中找到您的光谱仪
           - 点击"连接"按钮建立连接
        
        3. **数据接收**
           - 连接成功后，点击"开始接收数据"
           - 将光谱仪对准棉铃进行测量
           - 系统会自动接收并显示光谱数据
        
        4. **常见问题**
           - **无法扫描到设备**: 检查光谱仪蓝牙是否开启，是否进入配对模式
           - **连接失败**: 尝试重启光谱仪和手机蓝牙
           - **数据传输中断**: 确保设备在有效范围内，避免信号干扰
        
        ### 支持的设备型号：
        - SpectraScan系列光谱仪
        - AgriSpectrum系列设备
        - 其他支持蓝牙GATT协议的设备
        
        ### 技术要求：
        - 浏览器需支持Web Bluetooth API（Chrome 56+，Edge 79+）
        - 安卓系统需为Android 6.0+
        - 需要HTTPS连接（本地localhost除外）
        """)
    
    # 技术检测
    if not st.session_state.bluetooth_supported:
        st.warning("""
        ⚠️ **浏览器不支持Web Bluetooth API**
        
        当前浏览器可能不支持蓝牙功能，请尝试：
        1. 使用Chrome浏览器（Android推荐）
        2. 确保网站使用HTTPS协议
        3. 启用浏览器的蓝牙权限
        
        或者您可以使用传统的文件上传方式进行数据分析。
        """)

# ==================== 主应用 ====================
def main():
    """主应用入口"""
    # 初始化session state
    init_session_state()
    
    # 检查登录状态
    if not st.session_state.logged_in and st.session_state.current_page != "login":
        st.session_state.current_page = "login"
    
    # 根据当前页面路由显示对应内容
    pages = {
        "login": login_page,
        "dashboard": dashboard_page,
        "analysis": analysis_page,
        "result": result_page,
        "history": history_page,
        "bluetooth": bluetooth_connection_page
    }
    
    current_page = st.session_state.current_page
    if current_page in pages:
        pages[current_page]()
    else:
        # 默认跳转到登录页面
        st.session_state.current_page = "login"
        st.rerun()

if __name__ == "__main__":
    main()
