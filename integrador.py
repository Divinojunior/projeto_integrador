#pip freeze > requirements.txt (deletar pywin32==227)
#streamlit run integrador.py
# F1 'Convert Identations to Tab'

import os
import streamlit as st 
from PIL import Image
from datetime import date, time
import base64

# EDA Pkgs
import pandas as pd 
import numpy as np

# Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 

def main():
	""" Common ML Dataset Explorer """
	st.title("Projeto Integrador")
	
	html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Grupo 100Vies</p></div>
	"""
	
	st.markdown(html_temp,unsafe_allow_html=True)
	
	page = st.sidebar.selectbox("Escolha uma página", ["Visualização dos dados","Histórico de vendas","Predição por ML"])
	
	if page == "Visualização dos dados":
		st.header("Explore aqui o seu Dataset")
		visualize_data()
	elif page == "Histórico de vendas":
		st.title('Histórico de vendas')
		predicao_fernando()
	elif page == "Predição por ML":
		st.title('Prevendo a quantidade de peças')
		st_estimator = st.sidebar.number_input('Defina o n_estimator para o modelo Randon Forest Regressor [100, 500 ou 1000]: ', value=1000, min_value = 100, max_value = 1000, step=100)
		st_produto = st.sidebar.number_input('Escolha um Código de Produto [entre 1 e 201]: ', value=0, min_value = 0, max_value = 201, step=1)
		if st_produto != 0 and (st_estimator == 100 or st_estimator == 500 or st_estimator == 1000):
			predicao_mario(st_produto, st_estimator)

def visualize_data():

	def file_selector(folder_path='./Datasets'):
		filenames = os.listdir(folder_path)
		selected_filename = st.selectbox("Escolha um arquivo",filenames)
		return os.path.join(folder_path,selected_filename)

	filename = file_selector()
	st.info("Você selecionou {}".format(filename))

	
	# Ler dados
	df = pd.read_csv(filename)

	# Mostrar conjunto de dados
	if st.checkbox("Exibir dataset"):
		number = st.number_input("Número de linhas para exibir",1,200)
		st.dataframe(df.head(number))

	
	# Mostrar colunas
	#if st.button("Nomes das colunas"):
	#	st.write(df.columns)

	# Mostrar Forma
	if st.checkbox("Dimensões do dataset"):
		data_dim = st.radio("",("Linhas","Colunas"))
		if data_dim == 'Linhas':
			st.text("Numero de linhas")
			st.write(df.shape[0])
		elif data_dim == 'Colunas':
			st.text("Numero de colunas")
			st.write(df.shape[1])
		else:
			st.write(df.shape)

	# Select Columns
	if st.checkbox("Selecionar colunas a serem exibidas"):
		all_columns = df.columns.tolist()
		selected_columns = st.multiselect("Selecione",all_columns)
		new_df = df[selected_columns]
		st.dataframe(new_df)

	# Mostrar valores
	if st.button("Ranking de vendas"):
		st.text("Quantidade de vendas por produto")
		st.write(df.iloc[:,1].value_counts())
	
	# Mostrar tipos de dados
	if st.button("Tipos de dados das features"):
		st.write(df.dtypes)
	
	# Mostrar Resumo
	if st.checkbox("Estatística descritiva das features"):
		st.write(df.describe().T)

	# Gráfico e visualização
	st.subheader("Gráficos")
	# Correlação
	# Seaborn Plot
	if st.checkbox("Gráfico de Correlação [Seaborn]"):
		st.write(sns.heatmap(df.corr(),annot=True))
		st.pyplot()

	# Contagem
	if st.checkbox("Gráfico personalizável"):
		all_columns_names = df.columns.tolist()
		primary_col = st.selectbox("Agrupar por ",all_columns_names)
		selected_columns_names = st.multiselect("Feature a ser visualizada",all_columns_names)
		if st.button("Gráfico"):
			st.text("Gerar Gráfico")
			if selected_columns_names:
				vc_plot = df.groupby(primary_col)[selected_columns_names].count()
			else:
				vc_plot = df.iloc[:,-1].value_counts()
			st.write(vc_plot.plot(kind="bar"))
			st.pyplot()


	if st.button("Obrigado"):
		st.balloons()

def Teste (LinhaProduto,Codigo,CentroCusto,Cliente):

	import pandas as pd
	import plotly.express as px 

	#x = input('1-Vendas_Integrador_100Vies-Rev2.csv')
	#meu endereço - C:\Users\fgolo\Desktop\Integrador\1-Vendas_Integrador_100Vies-Rev2.csv

	df = pd.read_csv('./Datasets/1-Vendas_Integrador_100Vies-Rev2.csv')

	df['ano'] = df['DataVenda'].str[6:10]
	df['mes'] = df['DataVenda'].str[3:5]
	df['dia'] = df['DataVenda'].str[0:2]
	df['data'] = df['ano']+'-'+df['mes']+'-'+df['dia']
	df['data'] = pd.to_datetime(df['data'])
	df['semana'] = df['data'].dt.week
	df['semana'] = df['semana'].apply(str)
	df['semana'] = df['semana'].apply(lambda x: x.zfill(2))
	df['ano-semana'] = 'a' + df['ano'] + 's' + df['semana']  
	df = df.sort_values(by=['ano','mes','dia'])

	#df.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste.csv')

	def Teste2 (LinhaProduto):
		for i in [LinhaProduto]:
			if i in df.values:
				df2 = df.loc[df['LinhaProduto'] == i]
				#df2.to_csv(r'C:\Users\divin\Downloads\Teste2.csv')
			elif i == 'Todos':
				df2 = df
				#df2.to_csv(r'C:\Users\divin\Downloads\Teste2_todos.csv')
			else:
				print ("Linha de Produto não existente")
		return df2 

	df2 = Teste2(LinhaProduto)   
	
	def Teste3 (Codigo):
		for j in [Codigo]:
			if j in df2.values:
				df3 = df2.loc[df2['Codigo'] == j]
				#df3.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste3.csv')
			elif j == 'Todos':
				df3 = df2
				#df3.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste3.csv')
			else:
				print ("Codigo não pertence a Lista de Produtos selecionada")
		return df3

	df3 = Teste3(Codigo)

	def Teste4 (CentroCusto):
		for k in [CentroCusto]:
			if k in df3.values:
				df4 = df3.loc[df3['CentroCusto'] == k]
				#df4.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste4.csv')
			elif k == 'Todos':
				df4 = df3
				#df4.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste4.csv')
			else:
				print ("Centro de Custo não vendeu esse componente")
		return df4

	df4 = Teste4(CentroCusto)

	def Teste5 (Cliente):
		for l in [Cliente]:
			if l in df4.values:
				df5 = df4.loc[df4['Cliente'] == l]
				#df5.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste5.csv')
			elif l == 'Todos':
				df5 = df4
				#df5.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste5.csv')
			else:
				print ("Cliente não comprou desse centro de custo")
		return df5
	
	df5 = Teste5(Cliente)

	df6 = df5.groupby(['ano','ano-semana','Codigo','Cliente']).sum().reset_index()
	df6 = df6.sort_values(by=['ano'])

	def semanaa (a,b):
		semanaa = []
		for i in range(2017,b):
			i = str(i)
			for j in range(1,a):
				j = str(j)
				j = j.zfill(2)
				z = 'a' + i + 's' + j
				semanaa.append(z)
		return semanaa

	semanaa = semanaa(53,2021)

	def semana ():
		semana = df6['ano-semana'].to_list()
		for i in semanaa:
			if i not in semana:
				semana.append(i)
			else:
				pass
		return semana

	a = semana()

	def Teste6 (x):
		semana = []
		quantidade = []
		cliente = []
		z = 0
		for i in a:
			if i in df6.values:
				x = df6.at[z,'Quantidade']
				y = df6.at[z,'Cliente']
				semana.append(i)
				quantidade.append(x)
				cliente.append(y)
				z = z+1
			else:
				semana.append(i)
				quantidade.append(0)
				cliente.append ('')              
				z = z+1
		dfa = pd.DataFrame({'semana':semana,'cliente':cliente,'quantidade':quantidade})
		dfa = dfa.sort_values(by='semana')
		
		return dfa 
	
	dfa = Teste6(a)
	dfa = dfa.reset_index()
	dfaa = dfa.groupby(['semana']).sum().reset_index()
	dfaa = dfaa.sort_values(by='semana')
	dfa['semana'] = dfa['semana'].str[5:8] + dfa['semana'].str[0:5]
	suporte = dfa['semana'].to_list()
	
	#dfa.to_csv(r'C:\Users\fgolo\Desktop\Teste\TesteT.csv')
	#dfaa.to_csv(r'C:\Users\fgolo\Desktop\Teste\TesteTT.csv')

	fig = px.bar(dfa , x='semana' , y='quantidade' , color='cliente'  , category_orders={'semana':suporte},
				 template='plotly_dark' , color_discrete_sequence=px.colors.sequential.Rainbow)
	
	fig2 = px.line(dfaa , x='semana' , y='quantidade' , category_orders={'semana':suporte} ,
				   template='plotly_dark')

	st.write(fig)
	st.write(fig2)

def predicao_fernando():

	#df = pd.read_csv('./Datasets/1-Vendas_Integrador_100Vies-Rev2.csv')
	#st_linhaproduto = st.sidebar.multiselect('Selecione a linha do produto', df['LinhaProduto'].unique())
	#st_codproduto = st.sidebar.multiselect('Selecione o código do produto', df['Codigo'].unique())
	#st_cliente = st.sidebar.multiselect('Selecione o cliente', df['Cliente'].unique())


	if st.sidebar.checkbox("Selecionar linha do produto?",value=True):
		#st_linhaproduto = st.sidebar.number_input('Selecione a linha do produto', value=1,min_value = 1, max_value = 201, step=1)
		st_linhaproduto = st.sidebar.number_input('Selecione a linha do produto [entre 1 e 2]', value=0, min_value = 0, max_value = 201, step=1)
	else:
		st_linhaproduto = 'Todos'

	if st.sidebar.checkbox("Selecionar código do produto?",value=True):
		#st_codproduto = st.sidebar.text_input('Selecione o codigo do produto', value='A0000180133')
		st_codproduto = st.sidebar.number_input('Selecione o codigo do produto [entre 1 e 201]', value=0, min_value = 0, max_value = 201, step=1)
	else:
		st_codproduto = 'Todos'

	if st.sidebar.checkbox("Selecionar cliente?",value=True):
		#st_cliente = st.sidebar.number_input('Selecione o cliente', value=1, min_value = 1, max_value = 999, step=1)
		st_cliente = st.sidebar.number_input('Selecione o cliente', value = 0, min_value = 0, max_value = 999, step=1)
	else:
		st_cliente = 'Todos'

	#st_codproduto = st.sidebar.text_input('Selecione o codigo do produto')
	#st_cliente = st.sidebar.text_input('Selecione o cliente')

	#	if st_linhaproduto != '' and st_linhaproduto != 'Todos':
	#		st_linhaproduto == int(st_linhaproduto)
	#	elif st_codproduto != '' and st_codproduto != 'Todos':
	#		st_codproduto == int(st_codproduto)
	#	elif st_cliente != '' and st_cliente != 'Todos':
	#		st_cliente == int(st_cliente)

	if st_linhaproduto != 0 and st_codproduto != 0 and st_cliente != 0:
		#Teste('Todos','Todos',21320,'Todos')
		#Teste(1,'A0000180133',21320,1)
		#print ("Linha de Produto = ", st_linhaproduto)

		Teste(st_linhaproduto,st_codproduto,21320,st_cliente)

def predicao_mario(f_produto,f_estimator):
	#IMPORTANDO BIBLIOTECAS
	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)

	import matplotlib.pyplot as plt
	from sklearn import metrics
	from sklearn.model_selection import train_test_split
	import pandas as pd
	import seaborn as sns
	import numpy as np
	import plotly
	import os
	#import janitor


	from sklearn.metrics import mean_squared_log_error
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import mean_absolute_error as mae
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.linear_model import LinearRegression
	from sklearn.model_selection import cross_val_score
	from sklearn.model_selection import TimeSeriesSplit 
	from lightgbm import LGBMRegressor
	from sklearn.ensemble import RandomForestClassifier

	#from imblearn.over_sampling import SMOTE

	import statsmodels.formula.api as smf            # statistics and econometrics
	import statsmodels.tsa.api as smt
	import statsmodels.api as sm
	import scipy.stats as scs

	from itertools import product                    # some useful functions
	from tqdm import tqdm_notebook

	import warnings                                  # `do not disturbe` mode
	warnings.filterwarnings('ignore')

	def mean_absolute_percentage_error(y_true, y_pred): 
		return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

	pd.set_option('display.max_rows', None)
	params = {'legend.fontsize': 'x-large',
			'figure.figsize': (16, 8),
			'axes.labelsize': 'x-large',
			'axes.titlesize':'x-large',
			'xtick.labelsize':'x-large',
			'ytick.labelsize':'x-large'}

	#get_ipython().run_line_magic('matplotlib', 'inline')
	plt.rcParams.update(params)


	# In[3]:


	# # IMPORTAÇÃO DO DATASET DE VENDAS

	# In[4]:


	vendastelepweek=pd.read_csv("./Datasets/4-VendasTelepWeek-Rev3.csv",sep=',',encoding='Latin1')
	vendastelepweek.head()


	# In[5]:


	melt = vendastelepweek.melt(id_vars='Codigo', var_name='Week',value_name='Vendas')
	melt = melt.sort_values(['Week','Codigo']).reset_index(drop=True)
	melt['Week'] = melt['Week'].astype(int)
	melt = melt.sort_values(['Week','Codigo']).reset_index(drop=True)
	#melt.head(2)


	# # FEATURE ENGINEERING 1: MESES DO ANO

	# In[6]:


	Jan = []
	for w in melt['Week']:
		if (w<=4) or (w>=52 and w<=56)or (w>=104):
			jan = 1
		else:
			jan = 0
		Jan.append(jan)
	melt['Jan'] = Jan

	Fev = []
	for w in melt['Week']:
		if (w>=5 and w<=8) or (w>=57 and w<=60):
			fev = 1
		else:
			fev = 0
		Fev.append(fev)
	melt['Fev'] = Fev


	Mar = []
	for w in melt['Week']:
		if (w>=9 and w<=12) or (w>=61 and w<=64):
			mar = 1
		else:
			mar = 0
		Mar.append(mar)
	melt['Mar'] = Mar


	Abr = []
	for w in melt['Week']:
		if (w>=13 and w<=17) or (w>=65 and w<=69):
			abr = 1
		else:
			abr = 0
		Abr.append(abr)
	melt['Abr'] = Abr


	Mai = []
	for w in melt['Week']:
		if (w>=18 and w<=21) or (w>=70 and w<=73):
			mai = 1
		else:
			mai = 0
		Mai.append(mai)
	melt['Mai'] = Mai


	Jun = []
	for w in melt['Week']:
		if (w>=22 and w<=25) or (w>=74 and w<=77):
			jun = 1
		else:
			jun = 0
		Jun.append(jun)
	melt['Jun'] = Jun


	Jul = []
	for w in melt['Week']:
		if (w>=26 and w<=30) or (w>=78 and w<=82):
			jul = 1
		else:
			jul = 0
		Jul.append(jul)
	melt['Jul'] = Jul


	Ago = []
	for w in melt['Week']:
		if (w>=31 and w<=34) or (w>=83 and w<=86):
			ago = 1
		else:
			ago = 0
		Ago.append(ago)
	melt['Ago'] = Ago


	Set = []
	for w in melt['Week']:
		if (w>=35 and w<=38) or (w>=87 and w<=91):
			set = 1
		else:
			set = 0
		Set.append(set)
	melt['Set'] = Set

	Out = []
	for w in melt['Week']:
		if (w>=39 and w<=43) or (w>=92 and w<=95):
			out = 1
		else:
			out = 0
		Out.append(out)
	melt['Out'] = Out


	Nov = []
	for w in melt['Week']:
		if (w>=44 and w<=47) or (w>=96 and w<=99):
			nov = 1
		else:
			nov = 0
		Nov.append(nov)
	melt['Nov'] = Nov

	Dez = []
	for w in melt['Week']:
		if (w>=48 and w<=52) or (w>=100 and w<=104):
			dez = 1
		else:
			dez = 0
		Dez.append(dez)
	melt['Dez'] = Dez

	#melt.head()


	# # NORMALIZAÇÃO DA VARIÁVEL TARGET 'VENDAS'
	# ( VENDA - MEDIA ) / DESVIO --> PRODUTO A PRODUTO

	# ## CÁCULO DA MÉDIA E DESVIO PADRÃO DE CADA PRODUTO

	# In[7]:


	media = []
	desvio = []

	for var_produto in melt['Codigo']:
		med = melt[melt['Codigo'] == var_produto]['Vendas'].mean()
		media.append(med)
		
		dev = melt[melt['Codigo'] == var_produto]['Vendas'].std()
		desvio.append(dev)


	# In[8]:


	melt['Media'] = pd.Series(media)
	melt['Desvio'] = pd.Series(desvio)


	# ## NORMALIZAÇÃO DA VARIÁVEL TARGET 'VENDAS''
	# ( VENDA - MEDIA ) / DESVIO --> PRODUTO A PRODUTO

	# In[9]:


	melt['VendasStd'] = (melt['Vendas']-melt['Media'])/melt['Desvio']
	#melt.head()


	# # FEATURE ENGINEERING 2: LAG (1 - 4), DIFF (1 - 4)

	# In[10]:


	melt['VendasStd_Diff1'] = round(melt.groupby(['Codigo'])['VendasStd'].diff(),4)
	melt['VendasStd_Diff2'] = round(melt.groupby(['Codigo'])['VendasStd'].diff(2),4)
	melt['VendasStd_Diff3'] = round(melt.groupby(['Codigo'])['VendasStd'].diff(3),4)
	melt['VendasStd_Diff4'] = round(melt.groupby(['Codigo'])['VendasStd'].diff(4),4)

	melt['VendasStd_Lag1'] = round(melt.groupby(['Codigo'])['VendasStd'].shift(),4)
	melt['VendasStd_Lag2'] = round(melt.groupby(['Codigo'])['VendasStd'].shift(2),4)
	melt['VendasStd_Lag3'] = round(melt.groupby(['Codigo'])['VendasStd'].shift(3),4)
	melt['VendasStd_Lag4'] = round(melt.groupby(['Codigo'])['VendasStd'].shift(4),4)

	melt[melt['Codigo']==1].tail()


	# # BASELINE - LASTWEEK - LAG(1)
	# MAE = 0.66842

	# In[11]:


	melt['VendasStd_Lag1'] = melt.groupby(['Codigo'])['VendasStd'].shift()
	#melt.head(2)


	# In[12]:


	def rmse (ytrue,ypred):
		return np.sqrt(mean_squared_error(ytrue,ypred))


	# In[13]:


	mean_error = []
	for week in range(84,109):                #31/12/2019 - 24 semanas - 6 meses
		train = melt[melt['Week'] < week]
		val = melt[melt['Week'] == week]
		
		p = val['VendasStd_Lag1'].values
		
		error = mae(val['VendasStd'].values, p)
		#print('Week %d - Error %.5f' % (week, error))
		mean_error.append(error)
	#print('Mean Absolute Error = %.5f' % np.mean(mean_error))


	# In[14]:


	#prod= melt[melt['Codigo']==1]
	# multiple line plot
	#plt.plot( 'Week', 'VendasStd', data=prod, marker='', markerfacecolor='blue', markersize=12, color='green', linewidth=3,label="True")
	#plt.plot( 'Week', 'VendasStd_Lag1', data=prod, marker='', color='red', linewidth=1,label="Predict",linestyle='dashed')
	#plt.legend()
	#plt.title('BASELINE: Predict = VendasStd_Lag1\nMean Absolute Error = %.5f' % np.mean(mean_error))
	
	#st.pyplot(plt)

	# # 1o MODELO: RANDOM FOREST REGRESSOR
	# PRODUTO A PRODUTO
	# 
	# MAE: 0.30068

	# In[15]:


	def timeseries_train_test_split(X, y, test_size):
		"""
			Perform train-test split with respect to time series structure
		"""
		
		# get the index after which test set starts
		test_index = int(len(X)*(1-test_size))
		
		X_train = X.iloc[:test_index]
		y_train = y.iloc[:test_index]
		X_test = X.iloc[test_index:]
		y_test = y.iloc[test_index:]
		
		return X_train, X_test, y_train, y_test


	#c = int(input('Escolha um Código de Produto [entre 1 e 201]: '))
	#st_produto = st.sidebar.number_input('Escolha um Código de Produto [entre 1 e 201]: ', value=0, min_value = 0, max_value = 201, step=1)
	var_produto = f_produto

	prod = melt[melt['Codigo'] == var_produto]
	prod = prod.drop(['Vendas'],axis=1)
	y = prod.dropna().VendasStd
	X = prod.dropna().drop(['VendasStd'], axis=1)
	X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

	#Random Forest Regressor:
	rfr = RandomForestRegressor(n_estimators=f_estimator, n_jobs=-1, random_state=0)
	rfr.fit(X_train, y_train)
	p_RFR = rfr.predict(X_test)
	MAE_RFR = mae(y_test, p_RFR)

	X_test_RFR = X_test.copy()
	X_test_RFR['VendasStd_True'] = pd.Series(y_test)
	X_test_RFR['PredictStd'] = p_RFR
	X_test_RFR['MAE'] = abs(X_test_RFR['VendasStd_True'] - X_test_RFR['PredictStd'])

	X_test_RFR['Vendas_True'] = (X_test_RFR['VendasStd_True']*X_test_RFR['Desvio'])+X_test_RFR['Media']
	X_test_RFR['Predict_True'] = (X_test_RFR['PredictStd']*X_test_RFR['Desvio'])+X_test_RFR['Media']
	X_test_RFR['MAE_True'] = abs(X_test_RFR['Vendas_True'] - X_test_RFR['Predict_True'])

	#LGBM
	lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.01)
	lgbm.fit(X_train, y_train)
	p_LGBM = lgbm.predict(X_test)
	MAE_LGBM = mae(y_test,p_LGBM)

	X_test_LGBM = X_test.copy()
	X_test_LGBM['VendasStd_True'] = pd.Series(y_test)
	X_test_LGBM['PredictStd'] = p_LGBM
	X_test_LGBM['MAE'] = abs(X_test_LGBM['VendasStd_True'] - X_test_LGBM['PredictStd'])

	X_test_LGBM['Vendas_True'] = (X_test_LGBM['VendasStd_True']*X_test_LGBM['Desvio'])+X_test_LGBM['Media']
	X_test_LGBM['Predict_True'] = (X_test_LGBM['PredictStd']*X_test_LGBM['Desvio'])+X_test_LGBM['Media']
	X_test_LGBM['MAE_True'] = abs(X_test_LGBM['Vendas_True'] - X_test_LGBM['Predict_True'])

	# multiple line plot
	plt.plot( 'Week', 'Vendas_True', data=X_test_LGBM, marker='', markerfacecolor='blue', markersize=12, color='green', linewidth=3,label="True")
	plt.plot( 'Week', 'Predict_True', data=X_test_RFR, marker='', color='blue', linewidth=1,label="Predict_RFR",linestyle='dashed')
	plt.plot( 'Week', 'Predict_True', data=X_test_LGBM, marker='', color='red', linewidth=1,label="Predict_LGBM",linestyle='dashed')
	plt.legend()
	plt.title('Codigo %d - LGBM: Predict x True\n MAE RFR: %.5f\nMAE LGBM: %.5f' % (var_produto, MAE_RFR,MAE_LGBM))

	st.pyplot(plt)


	# # SHAP VALUES

	# In[ ]:


	import shap
	shap.initjs()

	#c = int(input('Escolha um Código de Produto [entre 1 e 201]: '))

	var_produto = f_produto

	prod = melt[melt['Codigo'] == var_produto]
	prod = prod.drop(['Vendas'],axis=1)
	y = prod.dropna().VendasStd
	X = prod.dropna().drop(['VendasStd'], axis=1)
	X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

	import shap
	#shap_values = shap.TreeExplainer(rfr).shap_values(X_train)
	#print('Codigo %d - Shap Value - Features Explainer' % (var_produto))
	#shap.summary_plot(shap_values, X_train, plot_type="bar")
	#shap.summary_plot(shap_values, X_train)
	#st.write(shap.summary_plot(shap_values, X_train, plot_type="bar"))
	#st.write(shap.summary_plot(shap_values, X_train))
	#st.pyplot.shap.summary_plot(shap_values, X_train, plot_type="bar")

if __name__ == '__main__':
	main()