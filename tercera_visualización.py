
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import numpy as np
import pandas as pd


st.set_page_config(page_title="Rendimiento académico - Contour", layout="centered")

st.title("RENDIMIENTO ACADÉMICO SEGÚN HORAS DE ESTUDIO Y SUEÑO")

st.write("""
Este análisis explora la relación entre las horas dedicadas al estudio y las horas de sueño en el rendimiento académico promedio de un grupo de estudiantes.
Los contornos representan niveles de rendimiento similares, mientras que los tonos más cálidos corresponden a mejores resultados.
El gráfico permite identificar los rangos combinados de estudio y descanso donde el rendimiento tiende a ser máximo.
""")

#Cargar el archivo .csv

file_path = 'Student_Performance.csv'

df = pd.read_csv(file_path)



from scipy.interpolate import griddata
import plotly.graph_objects as go
import numpy as np


study_col = "Hours Studied"
sleep_col = "Sleep Hours"
perf_col  = "Performance Index"

# Definir bins
study_bins = np.arange(df[study_col].min(), df[study_col].max() + 1, 1)
sleep_bins = np.arange(df[sleep_col].min(), df[sleep_col].max() + 1, 1)

df_binned = df.copy()
df_binned["study_bin"] = pd.cut(df_binned[study_col], bins=study_bins, include_lowest=True)
df_binned["sleep_bin"] = pd.cut(df_binned[sleep_col], bins=sleep_bins, include_lowest=True)

# 3) Tabla 2D de medias de rendimiento
pivot = (
    df_binned
    .groupby(["study_bin", "sleep_bin"], observed=True)[perf_col]
    .mean()
    .unstack()
)

# Centros de los intervalos
x_centers = pivot.index.map(lambda iv: iv.mid).to_numpy()
y_centers = pivot.columns.map(lambda iv: iv.mid).to_numpy()

Z = pivot.to_numpy()


contour = go.Contour(
    x=x_centers, y=y_centers, z=Z.T,
    contours=dict(
        coloring="heatmap",
        showlabels=True, labelfont=dict(size=12)
    ),
    colorbar=dict(title="Nota media"),
    hovertemplate="Horas estudio=%{x:.1f}<br>Horas sueño=%{y:.1f}<br>Nota media=%{z:.1f}<extra></extra>"
)



fig = go.Figure(data=[contour])
fig.update_layout(
    title="RENDIMIENTO ACADEMICO SEGÚN HORAS DE ESTUDIO Y SUEÑO",
    xaxis_title="Horas de Estudio",
    yaxis_title="Horas de Sueño"
)
fig.show()
with st.sidebar:
    st.info("Fuente de datos: Kaggle\n\n"
            "[Student Performance – Multiple Linear Regression](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)")

st.plotly_chart(fig, theme=None)

