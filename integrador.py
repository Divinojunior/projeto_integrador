#pip freeze > requirements.txt (deletar )
#streamlit run integrador.py

import os
import streamlit as st 
from PIL import Image
from datetime import date, time
import base64

# EDA Pkgs
import pandas as pd 

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
	
	page = st.sidebar.selectbox("Escolha uma página", ["Data Scientist","Business"])
	
	if page == "Data Scientist":
		st.header("Explore aqui o seu Dataset")
		visualize_data()
	elif page == "Business":
		st.title("Prevendo a quantidade de peças")
		predicao_fernando()

def visualize_data():

	def file_selector(folder_path='./Datasets'):
		filenames = os.listdir(folder_path)
		selected_filename = st.selectbox("Select A file",filenames)
		return os.path.join(folder_path,selected_filename)

	filename = file_selector()
	st.info("You Selected {}".format(filename))

	# Read Data
	df = pd.read_csv(filename)
	# Show Dataset

	if st.checkbox("Show Dataset"):
		number = st.number_input("Number of Rows to View")
		st.dataframe(df.head(number))

	# Show Columns
	if st.button("Column Names"):
		st.write(df.columns)

	# Show Shape
	if st.checkbox("Shape of Dataset"):
		data_dim = st.radio("Show Dimension By ",("Rows","Columns"))
		if data_dim == 'Rows':
			st.text("Number of Rows")
			st.write(df.shape[0])
		elif data_dim == 'Columns':
			st.text("Number of Columns")
			st.write(df.shape[1])
		else:
			st.write(df.shape)

	# Select Columns
	if st.checkbox("Select Columns To Show"):
		all_columns = df.columns.tolist()
		selected_columns = st.multiselect("Select",all_columns)
		new_df = df[selected_columns]
		st.dataframe(new_df)
	
	# Show Values
	if st.button("Value Counts"):
		st.text("Value Counts By Target/Class")
		st.write(df.iloc[:,-1].value_counts())


	# Show Datatypes
	if st.button("Data Types"):
		st.write(df.dtypes)



	# Show Summary
	if st.checkbox("Summary"):
		st.write(df.describe().T)

	## Plot and Visualization

	st.subheader("Data Visualization")
	# Correlation
	# Seaborn Plot
	if st.checkbox("Correlation Plot[Seaborn]"):
		st.write(sns.heatmap(df.corr(),annot=True))
		st.pyplot()

	
	# Pie Chart
	if st.checkbox("Pie Plot"):
		all_columns_names = df.columns.tolist()
		if st.button("Generate Pie Plot"):
			st.success("Generating A Pie Plot")
			st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
			st.pyplot()

	# Count Plot
	if st.checkbox("Plot of Value Counts"):
		st.text("Value Counts By Target")
		all_columns_names = df.columns.tolist()
		primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
		selected_columns_names = st.multiselect("Select Columns",all_columns_names)
		if st.button("Plot"):
			st.text("Generate Plot")
			if selected_columns_names:
				vc_plot = df.groupby(primary_col)[selected_columns_names].count()
			else:
				vc_plot = df.iloc[:,-1].value_counts()
			st.write(vc_plot.plot(kind="bar"))
			st.pyplot()


	# Customizable Plot

	st.subheader("Customizable Plot")
	all_columns_names = df.columns.tolist()
	type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
	selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

	if st.button("Generate Plot"):
		st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

		# Plot By Streamlit
		if type_of_plot == 'area':
			cust_data = df[selected_columns_names]
			st.area_chart(cust_data)

		elif type_of_plot == 'bar':
			cust_data = df[selected_columns_names]
			st.bar_chart(cust_data)

		elif type_of_plot == 'line':
			cust_data = df[selected_columns_names]
			st.line_chart(cust_data)

		# Custom Plot 
		elif type_of_plot:
			cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
			st.write(cust_plot)
			st.pyplot()

def Teste (LinhaProduto,Codigo,CentroCusto,Cliente):

    import pandas as pd
    import plotly.express as px 

    #x = input('1-Vendas_Integrador_100Vies-Rev1.csv')
    #meu endereço - C:\Users\fgolo\Desktop\Integrador\1-Vendas_Integrador_100Vies-Rev1.csv

    df = pd.read_csv('./Datasets/1-Vendas_Integrador_100Vies-Rev1.csv')
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
                #df2.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste2.csv')
            elif i == 'Todos':
                df2 = df
                #df2.to_csv(r'C:\Users\fgolo\Desktop\Teste\Teste2.csv')
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

    #fig.show()
    #fig2.show()
    st.write(fig)
    st.write(fig2)

def predicao_fernando():
    #st_linhaproduto = st.sidebar.number_input('Linha do produto', value=1, min_value = 0, max_value = 1000, step=1)
    #st_codigoproduto = st.sidebar.number_input('Código do produto', value=1, min_value = 0, max_value = 1000, step=1)
    #st_cliente = st.sidebar.number_input('Cliente', value=1, min_value = 0, max_value = 1000, step=1)
    
    Teste('Todos','Todos',21320,'Todos')
    #Teste(st_linhaproduto,st_codigoproduto,21320,st_cliente)
    
if __name__ == '__main__':
	main()