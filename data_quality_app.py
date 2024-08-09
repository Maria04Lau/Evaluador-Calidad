import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Función para evaluar la completitud de los datos
def evaluar_completitud(df):
    completitud = df.notnull().mean() * 100
    completitud_column = df.notnull().astype(int)
    return completitud, completitud_column

# Función para evaluar la unicidad de los datos
def evaluar_unicidad(df):
    unicidad = df.nunique() / len(df) * 100
    unicidad_column = df.apply(lambda col: col.duplicated(keep=False).astype(int).replace({1: 0, 0: 1}))
    return unicidad, unicidad_column

# Función para evaluar outliers en las columnas numéricas
def evaluar_outliers(df):
    outliers = {}
    outliers_column = pd.DataFrame()
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        non_outliers = df[col].between(lower_bound, upper_bound).mean() * 100
        outliers[col] = non_outliers
        outliers_column[col] = df[col].between(lower_bound, upper_bound).astype(int)
    return pd.Series(outliers), outliers_column

# Función principal para calcular la calidad de los datos
def calcular_calidad(df, dimensiones):
    calidad = pd.DataFrame(index=df.columns, columns=["Valores únicos", "Completitud (%)", "Unicidad (%)", "Outliers (%)", "Puntaje Total"])
    calidad["Valores únicos"] = [df[col].nunique() for col in df.columns]
    
    valid_df = df.copy()
    
    if "Completitud" in dimensiones:
        completitud, completitud_column = evaluar_completitud(df)
        calidad["Completitud (%)"] = completitud
        for col in df.columns:
            valid_df[f"{col}_completitud"] = completitud_column[col]
    
    if "Unicidad" in dimensiones:
        unicidad, unicidad_column = evaluar_unicidad(df)
        calidad["Unicidad (%)"] = unicidad
        for col in df.columns:
            valid_df[f"{col}_unicidad"] = unicidad_column[col]
    
    if "Outliers" in dimensiones:
        outliers, outliers_column = evaluar_outliers(valid_df)
        calidad["Outliers (%)"] = calidad.index.map(outliers).fillna(np.nan)
        for col in outliers_column.columns:
            valid_df[f"{col}_outliers"] = outliers_column[col]
    
    calidad["Puntaje Total"] = calidad[["Completitud (%)", "Unicidad (%)", "Outliers (%)"]].mean(axis=1, skipna=True)
    
    total_fila = {
        "Valores únicos": np.nan,
        "Completitud (%)": calidad["Completitud (%)"].mean() if "Completitud" in dimensiones else np.nan,
        "Unicidad (%)": calidad["Unicidad (%)"].mean() if "Unicidad" in dimensiones else np.nan,
        "Outliers (%)": calidad["Outliers (%)"].mean() if "Outliers" in dimensiones else np.nan,
        "Puntaje Total": calidad["Puntaje Total"].mean()
    }
    calidad.loc["Total"] = total_fila
    
    final_df = valid_df.copy()
    return calidad, final_df

# Interfaz de usuario de Streamlit
st.title('Evaluador de Calidad de Datos')

uploaded_file = st.file_uploader("Cargar un archivo CSV", type=["csv"])
if uploaded_file:
    sep = st.text_input("Ingrese el separador del archivo CSV (por defecto es ',')", value=",")
    try:
        with st.spinner('Cargando y analizando el archivo...'):
            df = pd.read_csv(uploaded_file, sep=sep)
            st.write(f"Tamaño del DataFrame cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
            st.dataframe(df)
            
            columnas_seleccionadas = st.multiselect(
                "Seleccione las columnas que desea evaluar:",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )

            st.subheader("Descripción dimensiones de calidad:")
            st.markdown("""
            - **Valores únicos:** Cantidad de valores únicos presentes en un conjunto de datos.
            - **Completitud (%):** Grado en que los datos están completos.
            - **Unicidad (%):** Proporción de datos únicos en una columna.
            - **Outliers (%):** Proporción de datos no considerados como outliers.
            """)
            
            dimensiones_seleccionadas = st.multiselect(
                "Seleccione las dimensiones de calidad que desea evaluar:",
                options=["Completitud", "Unicidad", "Outliers"],
                default=["Completitud", "Unicidad", "Outliers"]
            )

            if st.button("Calcular Calidad de los Datos"):
                df_seleccionado = df[columnas_seleccionadas]
                calidad, final_df = calcular_calidad(df_seleccionado, dimensiones_seleccionadas)

                st.subheader("Resultados:")
                st.dataframe(calidad)

                calidad_numeric = calidad.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
                
                st.subheader("Mapa de Calor de la Calidad de los Datos:")
                fig, ax = plt.subplots()
                sns.heatmap(calidad_numeric, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
                st.pyplot(fig)
                
                if len(dimensiones_seleccionadas) > 0:
                    metrics = ["Completitud (%)", "Unicidad (%)", "Outliers (%)"]
                    metrics = [metric for metric in metrics if metric.replace(" (%)", "") in dimensiones_seleccionadas]
                    totals = calidad.loc["Total", metrics].values
                    st.subheader("Radar de Calidad Total")
                    fig = px.line_polar(r=totals, theta=metrics, line_close=True)
                    fig.update_traces(fill='toself')
                    st.plotly_chart(fig)

                st.bar_chart(calidad["Puntaje Total"])
    
                if "Outliers" in dimensiones_seleccionadas:
                    st.subheader("Resultados de Outliers:")
                    st.bar_chart(calidad["Outliers (%)"])

                st.subheader("Descargar CSV con score_calidad")
                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar datos de validación como CSV",
                    data=csv,
                    file_name='validated_data.csv',
                    mime='text/csv',
                )
                
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
