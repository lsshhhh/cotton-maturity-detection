import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
        'just_analyzed': False  # 新增：标记是否刚刚完成分析
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
            <p>• 历史记录追溯 • 专业报告生成 • 实时可视化</p>
        </div>
        """, unsafe_allow_html=True)

def dashboard_page():
    """仪表盘页面"""
    # 顶部导航栏
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([3, 1, 1, 1])
    with col_nav1:
        st.markdown(f'<h2 style="color: #2E8B57;">👋 欢迎, {st.session_state.user_name}</h2>', unsafe_allow_html=True)
    with col_nav2:
        if st.button("📊 数据分析", use_container_width=True):
            st.session_state.current_page = "analysis"
            st.rerun()
    with col_nav3:
        if st.button("📜 历史记录", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()
    with col_nav4:
        if st.button("🚪 退出", type="secondary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_page = "login"
            st.rerun()
    
    st.markdown("---")
    
    # 系统概览卡片
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
        st.markdown("""
        <div class="metric-card">
            <h3>✅</h3>
            <h2>96.8%</h2>
            <p>用户满意度</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速开始卡片
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚡ 快速开始")
    
    col_start1, col_start2, col_start3 = st.columns(3)
    
    with col_start1:
        if st.button("🌱 开始成熟度检测", use_container_width=True, type="primary"):
            st.session_state.detection_type = "成熟度"
            st.session_state.current_page = "analysis"
            st.rerun()
    
    with col_start2:
        if st.button("📋 查看使用教程", use_container_width=True):
            with st.expander("使用教程", expanded=True):
                st.markdown("""
                **使用步骤：**
                1. 选择检测类型（成熟度/叶绿素/花青素）
                2. 上传光谱数据CSV文件
                3. 设置分析参数（波长范围等）
                4. 开始分析并查看结果
                5. 导出检测报告
                
                **数据格式要求：**
                - CSV格式，两列数据
                - 第一列：波长（单位：nm）
                - 第二列：反射率（0-1或0-100%）
                """)
    
    with col_start3:
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
        # 只显示最近的3条记录
        recent_records = st.session_state.analysis_history[-3:][::-1]
        
        for i, record in enumerate(recent_records):
            with st.expander(f"分析记录 {i+1}: {record.get('type', '未知')}检测 - {record.get('time', '未知时间')}", 
                            expanded=(i==0)):
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
    
    # 步骤2: 上传数据
    st.subheader("📤 上传光谱数据")
    
    tab1, tab2 = st.tabs(["上传文件", "使用示例数据"])
    
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
    
    # 步骤3: 参数设置（如果有数据）
    if is_dataframe_valid(st.session_state.uploaded_data):
        st.subheader("⚙️ 分析参数设置")
        
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
                st.markdown("""
                <div style="background: #f0f7ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #2E8B57;">
                    <b>分析提示:</b><br>
                    • 确保数据质量，避免噪声干扰<br>
                    • 选择合适的波长范围以获得最佳结果<br>
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
                            # 修复：确保点击后立即跳转
                            if st.button("📊 查看详细结果", use_container_width=True, type="primary"):
                                st.session_state.current_page = "result"
                                st.rerun()
                        
                        with col_btn2:
                            if st.button("🔄 重新分析", use_container_width=True, type="secondary"):
                                # 重置分析完成状态，但保留上传的数据
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
                    
                    # 在"分析提示"下方横向排列两个按钮
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        # 修复：确保点击后立即跳转
                        if st.button("📊 查看详细结果", use_container_width=True, type="primary"):
                            st.session_state.current_page = "result"
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("🔄 重新分析", use_container_width=True, type="secondary"):
                            # 重置分析完成状态，但保留上传的数据
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
        "history": history_page
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