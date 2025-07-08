import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import codigo_aux.analisis003 as analisis

def main():
    func003()
    

def func001():
    # se filtro del catalogo de claves las muertes que fueron causadas por diabetes
    # leer datos 
    df_datos = pd.read_csv("DEFUN_2023/CATALOGOS_DEFUN_2023/catalogos/causa_defuncion.CSV", encoding='utf-8')
    
    # filtrar datos
    df_filtrado = df_datos[df_datos['DESCRIP'].str.contains('diabetes', case=False, na=False)]
    
    # guardar datos
    df_filtrado.to_csv("pruebas/arreglo_datos/001_causa_defuncion_diabetes.csv", encoding="utf-8", index=False)
    df_filtrado.to_csv("pruebas/claves/clave_causa_muerte.csv", encoding="utf-8", index=False)

def func002():
    # utilizanco el catalogo de claves filtrado con las muertes causadas por diabetes, se establece un filtro para los registros de muertes en el 2023 por esta enfermedad
    
    # leer datos 
    df_claves = pd.read_csv("pruebas/arreglo_datos/001_clave_filtro_causa_muerte_diabetes.csv", encoding='utf-8')
    df_datos = pd.read_csv("DEFUN_2023/Registros de defunciones/conjunto_de_datos_defunciones_registradas_2023_csv.CSV", encoding='utf-8')
    
    # filtrar datos
    # obtener listado de datos de una columna específica
    lista_claves = df_claves['CVE'].tolist()
    df_filtrado = df_datos[df_datos['causa_def'].isin(lista_claves)]
    
    # guardar datos
    df_filtrado.to_csv("pruebas/arreglo_datos/002_datos_filtrados_causa_diabetes.csv", encoding="utf-8", index=False)
    
def func003():
    # generacion de graficos de barras para la cantidad de muertes por diabetes en el 2023 de acuerdo a ciertos criterios y analisis de correlacion de datos
    df_datos = pd.read_csv("pruebas/arreglo_datos/002/002_datos_filtrados_causa_diabetes.csv", encoding='utf-8')
    list_datos = [
        {
        "grupo": "edad_agru",
        "url_clave": "pruebas/claves/clave_edad_agrup.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_edad_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by age",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_edad_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":100
        },
        # {
        # "grupo": "sexo",  
        # "url_clave": "pruebas/claves/clave_sexo.csv",
        # "url_guardado": "pruebas/arreglo_datos/003/muertes_por_sexo_diabetes.csv",
        # "titulo": "Deaths by diabetes in 2023 by gender",
        # "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_sexo_diabetes.png",
        # "x": "CVE",
        # "filtro_cantidad_minima":100
        # },
        {
        "grupo": "edo_civil",
        "url_clave": "pruebas/claves/clave_estado_civil.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_est_civil_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by marital status",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_est_civil_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":100
        },
        {
        "grupo": "escolarida",
        "url_clave": "pruebas/claves/clave_escolaridad.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_nvl_esc_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by education level",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_nvl_esc_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":100
        },
        {
        "grupo": "cond_act",
        "url_clave": "pruebas/claves/clave_cond_eco.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_cond_act_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by economic status",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_cond_act_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":100
        },
        {
        "grupo": "ocupacion",
        "url_clave": "pruebas/claves/clave_ocupacion.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_ocupacion_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by occupation",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_ocupacion_diabetes.png",
        "x": "CVE",
        "filtro_cantidad_minima":500
        },
        {
        "grupo": "derechohab",
        "url_clave": "pruebas/claves/clave_derechohabiencia.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_derechohab_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by medical center",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_derechohab_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":500
        },
        {
        "grupo": "area_ur",
        "url_clave": "pruebas/claves/clave_area_urbana_rural.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_area_ur_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by urban area",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_area_ur_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":0
        },
        {
        "grupo": "tloc_regis",
        "url_clave": "pruebas/claves/calve_tamaño_localidad.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_tloc_regis_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by locality size",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_tloc_regis_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":0
        },
        {
        "grupo": "nacionalid",
        "url_clave": "pruebas/claves/clave_nacionalidad.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_nacionalid_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by nationality",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_nacionalid_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":0
        },
        {
        "grupo": "mes_ocurr",
        "url_clave": "pruebas/claves/clave_mes.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_mes_ocurr_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by month of occurrence",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_mes_ocurr_diabetes.png",
        "x": "DESCRIP",
        "filtro_cantidad_minima":0
        },
        {
        "grupo": "causa_def",
        "url_clave": "pruebas/claves/clave_causa_muerte.csv",
        "url_guardado": "pruebas/arreglo_datos/003/muertes_por_causa_def_diabetes.csv",
        "titulo": "Deaths by diabetes in 2023 by type of diabetes",
        "url_guardado_grafico": "pruebas/arreglo_datos/003/muertes_por_causa_def_diabetes.png",
        "x": "CVE",
        "filtro_cantidad_minima":1000
        },
        
        ]
    
    for datos in list_datos:
        agrupacionGrafica(df_datos, datos)
    
    #agrupacionGrafica(df_datos, list_datos[11])
    
def func004():
    
    # creacion de un data frame con las variables predictoras y objetivo
    # Load the data
    df_datos = pd.read_csv("pruebas/arreglo_datos/002/002_datos_filtrados_causa_diabetes.csv", encoding='utf-8')

    # Definir variables predictoras y objetivo
    # se agrupan de acuerdo con las posibles variables a analizar
    df_datos_riesgo = df_datos.groupby(['causa_def', 'cond_act', 'escolarida', 'edo_civil', 'derechohab', 'area_ur', 'edad_agru', 'ocupacion']).size().reset_index(name='frecuencia')
    
    df_datos_riesgo.to_csv('pruebas/arreglo_datos/004/prueba_agrupacion_variables_pred_obj.csv', encoding="utf-8", index=False)
    
    # se categoriza respecto a las frecuencias de las combinatorias de las variables predictoras
    df_datos_categoria = df_datos_riesgo.copy()
    
    dic_url_pesos = {
        'causa_def':'pruebas/arreglo_datos/003/muertes_por_causa_def_diabetes.csv', 
        'cond_act':'pruebas/arreglo_datos/003/muertes_por_cond_act_diabetes.csv', 
        'escolarida':'pruebas/arreglo_datos/003/muertes_por_nvl_esc_diabetes.csv', 
        'edo_civil':'pruebas/arreglo_datos/003/muertes_por_est_civil_diabetes.csv', 
        'derechohab':'pruebas/arreglo_datos/003/muertes_por_derechohab_diabetes.csv', 
        'area_ur':'pruebas/arreglo_datos/003/muertes_por_area_ur_diabetes.csv', 
        'edad_agru':'pruebas/arreglo_datos/003/muertes_por_edad_diabetes.csv', 
        'ocupacion':'pruebas/arreglo_datos/003/muertes_por_ocupacion_diabetes.csv'
    }

    # Revisar cada clave valor del diccionario dic_url_pesos
    for clave, url in dic_url_pesos.items():
        df_pesos = pd.read_csv(url, encoding='utf-8')
        # Agregar la columna 'peso' del dataframe df_pesos al dataframe df_datos_categoria
        str_peso = clave + "_peso"
        df_datos_categoria = df_datos_categoria.merge(df_pesos[['CVE', str_peso]], left_on=clave, right_on='CVE', how='left')
        df_datos_categoria.drop(columns=['CVE'], inplace=True)
    
    # calculo del promedio de los pesos 
    list_colum = ['causa_def_peso','cond_act_peso','escolarida_peso','edo_civil_peso','derechohab_peso','area_ur_peso','edad_agru_peso','ocupacion_peso']
    
    # Calcular el promedio de las columnas en list_colum para cada registro
    df_datos_categoria['pesos'] = df_datos_categoria[list_colum].mean(axis=1)
    
    # Crear la columna 'categoria' basada en los valores de 'pesos'
    df_datos_categoria['categoria'] = pd.cut(
        df_datos_categoria['pesos'],
        bins=[-float('inf'), 0.05, 0.1, 0.2, 0.4, float('inf')],
        labels=[0, 1, 2, 3, 4]
    )
    
    # Convertir la columna 'causa_def' a valores numéricos utilizando los dos últimos caracteres
    df_datos_categoria['causa_def'] = df_datos_categoria['causa_def'].apply(lambda x: int(x[-2:]) if pd.notnull(x) and x[-2:].isdigit() else 0)
    
    # guardamos los resultados
    df_datos_categoria.to_csv('pruebas/arreglo_datos/004/prueba_categorizacion_frecuencias.csv', encoding="utf-8", index=False)
    
    # obtenemos el total de frecuencias para cada categoria de riesgo
    df_datos_categoria_riesgo = df_datos_categoria.copy()
    df_datos_categoria_riesgo = df_datos_categoria_riesgo.groupby('categoria').size().reset_index(name='frecuencia')
    
    # se guardan los datos
    df_datos_categoria_riesgo.to_csv('pruebas/arreglo_datos/004/prueba_frecuencia_categoria.csv', encoding="utf-8", index=False)
    
    # Recursive Feature Elimination (RFE) para ayuda a elegir las variables más importantes.
    # Separar Variables Predictoras y Objetivo
    X = df_datos_categoria.drop(columns=["categoria"])  # Variables predictoras
    y = df_datos_categoria["categoria"]  # Variable objetivo

    # Convertir variables categóricas a numéricas
    X = pd.get_dummies(X)

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear árbol de decisión con pesos balanceados
    clf = DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=42)

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    # Predicción
    y_pred = clf.predict(X_test)
    
    # Medir la eficiencia de la predicción
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    
    # Guardar los resultados en un archivo CSV
    resultados = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, report.split()[10], report.split()[11], report.split()[12]],
        "Acceptable Range": ["0.7-1.0", "0.7-1.0", "0.7-1.0", "0.7-1.0"]
    }
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv('pruebas/arreglo_datos/004/resultados_eficiencia.csv', encoding="utf-8", index=False)
    
    # Guardar las predicciones en un archivo CSV
    pd.DataFrame(y_pred, columns=["prediccion"]).to_csv('pruebas/arreglo_datos/004/predicciones.csv', encoding="utf-8", index=False)


# ------------------- funciones auxiliares -------------------    

def agrupacionGrafica(df_datos:pd.DataFrame, datos:dict):
    print(datos["grupo"])
    ## agrupacion de datos
    df_est_civil = df_datos.groupby(datos["grupo"]).size().reset_index(name='frecuencia')
    ## recuperacion de etiquetas de est_civil
    df_etiqueta_est_civil = pd.read_csv(datos["url_clave"], encoding='utf-8')
    df_est_civil = df_est_civil.merge(df_etiqueta_est_civil, left_on=datos["grupo"], right_on='CVE')
    ## calculo de pesos de las variables
    str_peso = datos["grupo"] + "_peso"
    df_est_civil[str_peso] = df_est_civil['frecuencia'] / df_est_civil['frecuencia'].sum()
    
    ## guardar datos
    df_est_civil.to_csv(datos["url_guardado"], encoding="utf-8", index=False)
    ## eliminacion de valores con frecuencia menor a 100
    df_est_civil = df_est_civil[df_est_civil['frecuencia'] >= datos["filtro_cantidad_minima"]]
    ## grafico de barras
    ax = df_est_civil.plot(x=datos["x"], y='frecuencia', kind='bar', title=datos["titulo"], figsize=(10, 6))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(datos["url_guardado_grafico"])

def correlacionGrafica(df_datos:pd.DataFrame, datos:dict):
    ## agrupacion de datos
    df_est_civil = df_datos.groupby([datos["grupo"], datos["correlacion"]]).size().reset_index(name='frecuencia')
    ## recuperacion de etiquetas de est_civil
    df_etiqueta_est_civil = pd.read_csv(datos["url_clave"], encoding='utf-8')
    df_est_civil = df_est_civil.merge(df_etiqueta_est_civil, left_on=datos["grupo"], right_on='CVE')
    ## guardar datos
    df_est_civil.to_csv(datos["url_guardado"], encoding="utf-8", index=False)
    ## eliminacion de valores con frecuencia menor a 100
    df_est_civil = df_est_civil[df_est_civil['frecuencia'] >= datos["filtro_cantidad_minima"]]
    ## grafico de barras
    ax = df_est_civil.plot(x=datos["x"], y='frecuencia', kind='bar', title=datos["titulo"], figsize=(10, 6))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(datos["url_guardado_grafico"])

if __name__ == "__main__":
    main()

