import pandas as pd
import matplotlib.pyplot as plt

def graficaEdad(df_datos:pd.DataFrame):
    ## agrupacion de datos
    df_edad = df_datos.groupby('edad_agru').size().reset_index(name='frecuencia')
    ## recuperacion de etiquetas de edad
    df_etiqueta_edad = pd.read_csv("pruebas/claves/clave_edad_agrup.csv", encoding='utf-8')
    df_edad = df_edad.merge(df_etiqueta_edad, on='edad_agru')
    ## guardar datos
    df_edad.to_csv("pruebas/arreglo_datos/003/muertes_por_edad_diabetes.csv", encoding="utf-8", index=False)
    ## eliminacion de valores con frecuencia menor a 100
    df_edad = df_edad[df_edad['frecuencia'] >= 100]
    ## grafico de barras
    ax = df_edad.plot(x='DESCRIP', y='frecuencia', kind='bar', title='Muertes por diabetes en 2023 por edad')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("pruebas/arreglo_datos/003/muertes_por_edad_diabetes.png")
    
def graficaSexo(df_datos:pd.DataFrame):
    ## agrupacion de datos
    df_sexo = df_datos.groupby('sexo').size().reset_index(name='frecuencia')
    ## recuperacion de etiquetas de sexo
    df_etiqueta_sexo = pd.read_csv("pruebas/claves/clave_sexo.csv", encoding='utf-8')
    df_sexo = df_sexo.merge(df_etiqueta_sexo, left_on='sexo', right_on='CVE')
    ## guardar datos
    df_sexo.to_csv("pruebas/arreglo_datos/003/muertes_por_sexo_diabetes.csv", encoding="utf-8", index=False)
    ## eliminacion de valores con frecuencia menor a 100
    df_sexo = df_sexo[df_sexo['frecuencia'] >= 100]
    ## grafico de barras
    ax = df_sexo.plot(x='sexo_y', y='frecuencia', kind='bar', title='Muertes por diabetes en 2023 por sexo')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("pruebas/arreglo_datos/003/muertes_por_sexo_diabetes.png")

def graficaEstadoCivil(df_datos:pd.DataFrame):
    ## agrupacion de datos
    df_est_civil = df_datos.groupby('edo_civil').size().reset_index(name='frecuencia')
    ## recuperacion de etiquetas de est_civil
    df_etiqueta_est_civil = pd.read_csv("pruebas/claves/clave_estado_civil.csv", encoding='utf-8')
    df_est_civil = df_est_civil.merge(df_etiqueta_est_civil, left_on='edo_civil', right_on='CVE')
    ## guardar datos
    df_est_civil.to_csv("pruebas/arreglo_datos/003/muertes_por_est_civil_diabetes.csv", encoding="utf-8", index=False)
    ## eliminacion de valores con frecuencia menor a 100
    df_est_civil = df_est_civil[df_est_civil['frecuencia'] >= 100]
    ## grafico de barras
    ax = df_est_civil.plot(x='DESCRIP', y='frecuencia', kind='bar', title='Muertes por diabetes en 2023 por estado civil', figsize=(10, 6))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("pruebas/arreglo_datos/003/muertes_por_est_civil_diabetes.png")
    
def graficaNivelEscolar(df_datos:pd.DataFrame):
    ## agrupacion de datos
    df_nvl_esc = df_datos.groupby('escolarida').size().reset_index(name='frecuencia')
    ## recuperacion de etiquetas de nvl_esc
    df_etiqueta_nvl_esc = pd.read_csv("pruebas/claves/clave_escolaridad.csv", encoding='utf-8')
    df_nvl_esc = df_nvl_esc.merge(df_etiqueta_nvl_esc, left_on='escolarida', right_on='CVE')
    ## guardar datos
    df_nvl_esc.to_csv("pruebas/arreglo_datos/003/muertes_por_nvl_esc_diabetes.csv", encoding="utf-8", index=False)
    ## eliminacion de valores con frecuencia menor a 100
    df_nvl_esc = df_nvl_esc[df_nvl_esc['frecuencia'] >= 100]
    ## grafico de barras
    ax = df_nvl_esc.plot(x='DESCRIP', y='frecuencia', kind='bar', title='Muertes por diabetes en 2023 por escolaridad', figsize=(10, 6))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("pruebas/arreglo_datos/003/muertes_por_nvl_esc_diabetes.png")