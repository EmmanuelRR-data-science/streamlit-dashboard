import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import os
import gdown

# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de Clientes RFM",
    page_icon="📈",
    layout="wide"
)

# --- URLs públicas de los archivos en Google Drive ---
URL_RFM = "https://drive.google.com/uc?id=1TPcjyw8Iok3ckYf9WgdosWI5Cm-h6ya8"
URL_MERGED = "https://drive.google.com/uc?id=1T-RzhM6VQDcSD2E9hbs5aYcUNu2uP-5b"

# --- Función para descargar archivos si no existen ---
def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Descargando {filename}..."):
            gdown.download(url, filename, quiet=False)

# Descargar los archivos CSV si es necesario
download_file(URL_RFM, "rfm_segmentation.csv")
download_file(URL_MERGED, "df_merged_with_segments.csv")

# --- Carga de datos ---
@st.cache_data
def load_data():
    df_rfm = pd.read_csv('rfm_segmentation.csv', index_col='Customer ID')
    df_merged = pd.read_csv('df_merged_with_segments.csv')
    return df_rfm, df_merged

try:
    df_rfm, df_merged = load_data()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# --- Dashboard ---

st.title("📈 Análisis del Comportamiento del Cliente y Oportunidades de Crecimiento")
st.markdown("""
Este dashboard interactivo presenta un análisis exploratorio del comportamiento de clientes de una tienda online.
El objetivo es identificar y caracterizar a los clientes más valiosos y a aquellos en riesgo de abandono para proponer estrategias de retención.
""")

# 1. Distribución de clientes por segmento RFM
st.subheader("1. Distribución de Clientes por Segmento RFM")
segment_counts = df_rfm['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segmento', 'Número de Clientes']
fig1 = px.bar(segment_counts, x='Segmento', y='Número de Clientes', 
             title='Número de Clientes por Segmento', color='Segmento')
st.plotly_chart(fig1, use_container_width=True)
st.markdown("""
_**Conclusión de la sección:**_ Este gráfico nos da una visión general de la base de clientes, mostrando qué porcentaje de nuestros clientes se encuentran en los segmentos más valiosos ("Loyal Customers") y cuáles están en riesgo de abandono ("At Risk"). Esta segmentación es fundamental para enfocar los esfuerzos de marketing de manera estratégica.
""")

# 2. Análisis de ventas por país (Mapa)
st.subheader("2. Análisis Geográfico de Clientes")
sales_by_country = df_merged.groupby('Country')['TotalPrice'].sum().reset_index()
fig2 = px.choropleth(sales_by_country, 
                    locations='Country', 
                    locationmode='country names',
                    color='TotalPrice',
                    hover_name='Country',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title='Ventas Totales por País')
st.plotly_chart(fig2, use_container_width=True)
st.markdown("""
_**Conclusión de la sección:**_ La mayor parte de las ventas proviene del Reino Unido, lo que lo confirma como nuestro mercado principal. Sin embargo, el mapa también destaca la presencia en otros países europeos, lo que representa una oportunidad para expandir y personalizar estrategias de marketing a nivel regional.
""")

# 3. Exploración detallada de los segmentos
st.subheader("3. Exploración Detallada de los Segmentos")

selected_segment = st.selectbox(
    "Selecciona un segmento para analizar:",
    df_rfm['Segment'].unique()
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Top 10 Productos del segmento '{selected_segment}'**")
    df_segment = df_merged[df_merged['Segment'] == selected_segment]
    top_products = df_segment['Description'].value_counts().head(10).reset_index()
    top_products.columns = ['Producto', 'Cantidad']
    st.dataframe(top_products, use_container_width=True)
    
with col2:
    st.markdown(f"**Comparación con el segmento 'Loyal Customers'**")
    loyal_df = df_merged[df_merged['Segment'] == 'Loyal Customers']
    top_products_loyal = loyal_df['Description'].value_counts().head(10).reset_index()
    top_products_loyal.columns = ['Producto', 'Cantidad']
    comparison_df = top_products.merge(top_products_loyal, on='Producto', how='outer', suffixes=(f' ({selected_segment})', ' (Loyal Customers)'))
    comparison_df = comparison_df.fillna(0)
    st.dataframe(comparison_df, use_container_width=True)

st.markdown("""
_**Conclusión de la sección:**_ Al comparar los productos, notamos una diferencia clave en el comportamiento de compra. Los "Loyal Customers" tienden a comprar productos más funcionales y de uso diario, mientras que los "At Risk" se enfocan en artículos decorativos. La oportunidad de negocio reside en promocionar los productos funcionales a los clientes en riesgo para convertirlos en clientes leales.
""")

# 4. Análisis de la canasta de productos (reglas de asociación)
st.subheader("4. Análisis de la Canasta de Productos")
st.markdown("Identifica productos que se compran juntos con frecuencia para estrategias de venta cruzada.")

@st.cache_data
def get_basket_analysis(dataframe):
    basket = (dataframe[dataframe['Country'] == 'United Kingdom']
              .groupby(['Invoice', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('Invoice'))
    def encode_units(x):
        return 1 if x >= 1 else 0
    basket_sets = basket.applymap(encode_units)
    frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules.sort_values('lift', ascending=False, inplace=True)
    return rules

rules = get_basket_analysis(df_merged)

st.markdown("### Reglas de Asociación (Top 10)")
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10), use_container_width=True)
st.markdown("""
_**Conclusión de la sección:**_ Las reglas de asociación revelan patrones de compra, como "los clientes que compraron A, también compraron B". Con esta información, podemos optimizar la ubicación de los productos en la tienda virtual o crear ofertas de paquetes para impulsar las ventas cruzadas y aumentar el valor de vida del cliente.
""")

st.markdown("---")
st.markdown("Este dashboard es una muestra del análisis de datos. Los resultados y visualizaciones se pueden usar para informar decisiones estratégicas de negocio.")
