import pandas as pd     # 資料處理函式庫，用於 操作 DataFrame 結構
import yfinance as yf   # Yahoo Finance 資料下載庫，用於 取得 股票歷史數據
import matplotlib.pyplot as plt  # 基礎繪圖函式庫，用於 繪製 趨勢或時間序列 的圖表 (線圖, 散點圖...)
import seaborn as sns   # 高階統計視覺化函式庫，用於 美化圖表 和 繪製 統計 圖表 (分佈圖, 相關性圖...)

# 設定 Seaborn ，讓圖表更專業
sns.set_style("whitegrid")  # 圖表風格設定

# 設定中文字體，解決中文亂碼問題
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'sans-serif']
# 解決負號可能出現的方塊問題
plt.rcParams['axes.unicode_minus'] = False
print("中文字體設定完成。")

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    從 Yahoo Finance 下載 股票歷史資料

    參數:
    ticker (str): 股票代號 (例如: 'SPY' 或 'AAPL')
    start_date (str): 開始日期 (格式: 'YYYY-MM-DD')
    end_date (str): 結束日期 (格式: 'YYYY-MM-DD')
    
    回傳:
    pd.DataFrame: 包含股價歷史數據的 DataFrame
    """

    print(f"正在下載 {ticker} 從 {start_date} 到 {end_date} 的資料...")

    try:
        # auto_adjust=False 為保留 Adj Close 欄位
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            raise ValueError(f"未能取得 {ticker} 的資料，請檢查 股票代號 和 日期範圍 是否正確。")
        return data
    except Exception as e:
        print(f"下載資料時發生錯誤: {e}")
        return pd.DataFrame()   # 回傳空的 DataFrame 作為錯誤處理
    
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    進行 資料清洗 和 特徵工程，計算關鍵指標。
    
    參數:
    df (pd.DataFrame): 原始股價數據
    
    回傳:
    pd.DataFrame: 包含新特徵的數據
    """

    # 確保索引是日期時間格式
    df.index = pd.to_datetime(df.index)

    # 計算日收益率 (Daily Return)
    df['Daily Return'] = df['Adj Close'].pct_change() * 100

    # 計算移動平均線 (Moving Averages) - 兩個常用的技術指標
    # 20日移動平均線 (SMA_20)
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    # 50日移動平均線 (SMA_50)
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()

    # 計算日內的波動幅度 (Range)
    df['Daily Range'] = df['High'] - df['Low']

    # 移除因為計算移動平均線產生的 NaN 值
    df.dropna(inplace=True)

    print(f"完成特徵工程，新增欄位: 'Daily Return', 'SMA_20', 'SMA_50', 'Daily Range'。")
    print(f"資料集目前有 {df.shape[0]} 筆資料和 {df.shape[1]} 個欄位。")
    return df

def visualize_data(df: pd.DataFrame, ticker: str):
    """
    使用 Matplotlib 和 Seaborn 繪製股價走勢圖 和 日收益率分佈圖。
    
    參數:
    df (pd.DataFrame): 包含 股價數據 和 特徵 的 DataFrame
    ticker (str): 股票代號
    """

    print("--- 3. 繪製視覺化圖表... ---")

    # 創建一個 2x2 的子圖佈局
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    # hspace (Height space): 調整子圖之間的垂直間距 (增加間距，防止重疊)
    fig.subplots_adjust(hspace=0.35, wspace=0.2)
    plt.suptitle(f'{ticker} 股價探索性資料分析 (EDA)', fontsize=16)

    # 圖表 1: 調整後的收盤價和移動平均線
    axes[0, 0].plot(df.index, df['Adj Close'], label='Adj Close', color='blue', linewidth=1.5)
    axes[0, 0].plot(df.index, df['SMA_20'], label='SMA 20', color='orange', linestyle='--')
    axes[0, 0].plot(df.index, df['SMA_50'], label='SMA 50', color='green', linestyle='--')
    axes[0, 0].set_title(f'{ticker} 調整後收盤價與移動平均線', fontsize=12)
    axes[0, 0].set_xlabel('日期')
    axes[0, 0].set_ylabel('價格 (USD)')
    axes[0, 0].legend()

    # 圖表 2: 日收益率分佈 (使用 Seaborn 的直方圖)
    sns.histplot(df['Daily Return'], bins=50, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title(f'{ticker} 日收益率分佈', fontsize=12)
    axes[0, 1].set_xlabel('日收益率 (%)')
    axes[0, 1].set_ylabel('頻率')

    # 圖表 3: 日內波動幅度 (Range) 隨時間的變化
    axes[1, 0].plot(df.index, df['Daily Range'], color='purple', alpha=0.7)
    axes[1, 0].set_title(f'{ticker} 日內波動幅度', fontsize=12)
    axes[1, 0].set_xlabel('日期')
    axes[1, 0].set_ylabel('價格範圍 (USD)')

    # 圖表 4: 交易量 (Volume) 趨勢
    axes[1, 1].plot(df.index, df['Volume'], color='grey', alpha=0.6)
    axes[1, 1].set_title(f'{ticker} 交易量趨勢', fontsize=12)
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('交易量')

    plt.show()
    print("圖表繪製完成。")

def main():
    """主執行函數，設定參數並依序執行步驟。"""

    # 可調整參數 --------------------------------------
    STOCK_TICKER = 'SPY'  # S&P 500 ETF 
    START_DATE = '2020-01-01'
    END_DATE = '2024-12-31'
    # ------------------------------------------------

    # 步驟 1: 下載股票數據
    stock_df = fetch_stock_data(STOCK_TICKER, START_DATE, END_DATE)

    if stock_df.empty:
        print("無法進行後續分析，程式結束。")
        return
    
    # 步驟 2: 特徵工程
    process_df = feature_engineering(stock_df)

    # 步驟 3: 基礎統計摘要
    print("\n--- 4. 基礎統計摘要 ---")
    print(process_df[['Adj Close', 'Daily Return', 'Daily Range']].describe())

    # 步驟 4: 視覺化數據
    visualize_data(process_df, STOCK_TICKER)

    print("\n--- 專案 股票探索性資料分析 (EDA) 完成！請查看產生的圖表 ---")

if __name__ == "__main__":
    main()