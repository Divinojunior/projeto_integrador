#import os
import streamlit as st 

# EDA Pkgs
#import pandas as pd 

# Viz Pkgs
#import matplotlib.pyplot as plt 
#import matplotlib
#matplotlib.use('Agg')
#import seaborn as sns 

def main():
	""" Common ML Dataset Explorer """
	st.title("Projeto Integrador")
	
	#html_temp = """
	#<div style="background-color:tomato;"><p style="color:white;font-size:50px;padding:10px">Grupo 100Vies</p></div>
	#"""
	
	#st.markdown(html_temp,unsafe_allow_html=True)
	
	page = st.sidebar.selectbox("Escolha uma página", ["Business" , "Data Scientist"])
	
	if page == "Data Scientist":
		st.header("Explore aqui o seu Dataset")
		#visualize_data()
	elif page == "Business":
		st.title("Calcule a quantidade de peças a serem compradas")


#def predicao():

if __name__ == '__main__':
	main()