# %%
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from pprint import pprint
import pandas as pd
import json
from crewai_tools import SerperDevTool
load_dotenv()


# %%
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
tavily_api_key = os.environ.get('TAVILY_API_KEY')
SERPER_API_KEY = os.environ.get('SERPER_API_KEY')
id_aplicación = os.environ.get('ID_APLICACION')
graph_token = os.environ.get('GRAPH_TOKEN')

# %%
model ='groq/llama-3.1-8b-instant'

# %%
llm_genmini_2_5 = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20', google_api_key=GOOGLE_API_KEY, temperature=0 )
llm_genmini_2_0 = ChatGroq(model=model, groq_api_key=os.environ['GROQ_API_KEY'], temperature=0, timeout=120 )

# %%

financial_dates_json = '''
[
    {
        "fecha_inicio": "2025-05-12",
        "fecha_fin": "2025-07-22",
        "evento": "Declaración de Renta - Personas Jurídicas",
        "tipo": "Tributario",
        "detalle": "Plazo escalonado según último dígito del NIT"
    },
    {
        "fecha_inicio": "2025-08-12",
        "fecha_fin": "2025-10-24",
        "evento": "Declaración de Renta - Personas Naturales",
        "tipo": "Tributario",
        "detalle": "Segmentado por cédula (Art. 590 Estatuto Tributario)"
    },
    {
        "fecha": "2025-06-30",
        "evento": "Primer pago de prima de servicios",
        "tipo": "Laboral",
        "detalle": "Obligatorio para empleadores (Código Sustantivo del Trabajo Art. 306)"
    },
    {
        "fecha": "2025-12-20",
        "evento": "Segundo pago de prima de servicios + prima de Navidad",
        "tipo": "Laboral",
        "detalle": "Pago máximo hasta esta fecha"
    },
    {
        "fecha": "2025-04-17",
        "evento": "Jueves Santo - Cierre operaciones financieras",
        "tipo": "Festivo operativo",
        "detalle": "Suspensión de actividades en entidades bancarias"
    },
    {
        "fecha": "2025-07-15",
        "evento": "Corte impuesto predial (general)",
        "tipo": "Municipal",
        "detalle": "Fecha referencial - varía por municipio"
    },
    {
        "fecha_inicio": "2025-01-01",
        "fecha_fin": "2025-03-31",
        "evento": "Actualización RUT (Registro Único Tributario)",
        "tipo": "Obligación fiscal",
        "detalle": "Plazo para actualizar datos en DIAN"
    },
    {
        "fecha": "2025-05-31",
        "evento": "Entrega estados financieros a Superintendencia de Sociedades",
        "tipo": "Corporativo",
        "detalle": "Informes consolidados para empresas reguladas"
    },
    {
        "fecha": "2025-11-28",
        "evento": "Pago última cuota de predial",
        "tipo": "Municipal",
        "detalle": "Para planes de pago fraccionado"
    },
    {
        "fecha": "2025-09-30",
        "evento": "Declaración IVA tercer trimestre",
        "tipo": "Tributario",
        "detalle": "Plazo general para régimen común"
    }
]
'''

# Crear DataFrame
df_finanzas = pd.read_json(financial_dates_json)

# Convertir fechas y ordenar
date_cols = ['fecha', 'fecha_inicio', 'fecha_fin']
for col in date_cols:
    if col in df_finanzas.columns:
        df_finanzas[col] = pd.to_datetime(df_finanzas[col])

df_finanzas = df_finanzas.sort_values(by='fecha_inicio', na_position='first')



# %%
festivos_data = {
    "Fecha": [
        "2025-01-01", "2025-01-06", "2025-03-24", "2025-04-17", 
        "2025-04-18", "2025-05-01", "2025-06-02", "2025-06-23", 
        "2025-06-30", "2025-07-20", "2025-08-07", "2025-08-18", 
        "2025-10-13", "2025-11-03", "2025-11-17", "2025-12-08", 
        "2025-12-25"
    ],
    "Festivo": [
        "Año Nuevo", "Reyes Magos", "Día de San José", "Jueves Santo", 
        "Viernes Santo", "Día del Trabajo", "Ascensión de Jesús", 
        "Corpus Christi", "San Pedro y San Pablo", "Independencia de Colombia", 
        "Batalla de Boyacá", "Asunción de la Virgen", "Día de la Raza", 
        "Todos los Santos", "Independencia de Cartagena", 
        "Inmaculada Concepción", "Navidad"
    ],
    "Tipo": [
        "Fijo", "Trasladable", "Trasladable", "Religioso", 
        "Religioso", "Internacional", "Religioso", "Religioso", 
        "Religioso", "Patrio", "Patrio", "Religioso", 
        "Cultural", "Religioso", "Patrio", "Religioso", "Religioso"
    ]
}

df_festivos = pd.DataFrame(festivos_data)
df_festivos["Fecha"] = pd.to_datetime(df_festivos["Fecha"])

# %%
search = SerperDevTool()

# %%
@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Used to process content found on the internet."""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

tools = [SerperDevTool(), process_search_tool]


# %%

# 1. Agente: Investigador de Tendencias Sociales
investigador_tendencias = Agent(
    role="Rastreador de virales en finanzas personales",
    goal="Identificar tendencias actuales en Instagram, Facebook y TikTok relacionadas con finanzas personales en Colombia",
    backstory="""Eres un experto en detectar contenido viral. Usas técnicas avanzadas de scraping y análisis de hashtags 
    para encontrar qué temas de ahorro, inversión o manejo de deudas están generando más engagement. Conoces los influencers 
    financieros clave en el mercado colombiano y monitoreas sus publicaciones en tiempo real. Tu especialidad es distinguir 
    entre modas pasajeras y tendencias sostenibles.""",
    tools=[search],
    verbose=True,
    allow_delegation=True,
    llm=llm_genmini_2_0,
    
)

# 2. Agente: Analista de Contexto Local
analista_contexto = Agent(
    role="Especialista en calendarios y cultura colombiana",
    goal="Integrar fechas relevantes (navidad, declaración de renta) al plan de contenido",
    backstory="""Tienes un profundo conocimiento del calendario fiscal colombiano y las festividades culturales. Sabes 
    exactamente cuándo los usuarios necesitan consejos para declarar renta (agosto), cómo aprovechar la prima (diciembre, junio). 
    Tu trabajo es asegurar que el contenido sea hiperrelevante según la época del año.""",
    tools=[search],
    verbose=True,
    allow_delegation=False,
    llm=llm_genmini_2_0
)

# 3. Agente: Estratega de Contenidos Multiplataforma
estratega_contenidos = Agent(
    role="Diseñador de contenido adaptativo",
    goal="Crear posts específicos para cada red social, optimizando horarios y formato",
    backstory="""Eres un maestro en psicología del engagement. Sabes que en TikTok se necesitan tips en formato de 15 segundos 
    con música trendy, mientras que en Facebook funcionan mejor guías descargables para audiencias mayores. Dominas los picos 
    de actividad: mañanas para consejos rápidos, tardes para tutoriales profundos, y noches para testimonios inspiradores.""",
    tools=[search],
    verbose=True,
    allow_delegation=True,
    llm=llm_genmini_2_0
)

# 4. Agente: Generador de Parrillas Inteligentes
generador_parrillas = Agent(
    role="Arquitecto de calendarios automatizados",
    goal="Convertir las ideas en un archivo Excel listo para publicar",
    backstory="""Eres un obsesivo del orden. Transformas ideas creativas en estructuras claras: columnas para fechas, 
    plataformas, copywriting, hashtags sugeridos y métricas esperadas. Tu Excel incluye fórmulas automáticas para calcular 
    días hábiles y evita fechas conflictivas.""",
    tools=[],  # No necesita herramientas de búsqueda
    verbose=True,
    allow_delegation=False,
    llm=llm_genmini_2_0
)

# %%
# Tarea para el Agente: Investigador de Tendencias Sociales
# Objetivo: Identificar tendencias en redes sociales sobre finanzas personales en Colombia.
# Dependencias: Ninguna inicial. Usa SerperDevTool y process_search_tool.
task_investigar_tendencias = Task(
    description=f"""
    Utiliza la herramienta 'Search the internet with Serper' para encontrar información sobre las tendencias actuales,
    hashtags populares y tipos de contenido que están generando más engagement (likes, comentarios,
    compartidos) en Instagram, Facebook y TikTok en COLOMBIA, específicamente en el nicho de
    'Finanzas personales', 'Ahorro', 'Inversión', 'Manejo de deudas', 'Educación financiera' u otros
    términos relacionados y populares localmente.

    Cuando uses la herramienta 'Search the internet with Serper', asegúrate de que el parámetro 'search_query'
    sea una **cadena de texto simple** que contenga la consulta de búsqueda, por ejemplo:
    `search_query='Tendencias actuales en finanzas personales en redes sociales colombianas'`

    Busca artículos de blog, noticias, reportes de tendencias digitales o cualquier fuente confiable
    que hable sobre lo viral en este tópico en redes sociales colombianas.
    Utiliza 'process_search_tool' para leer el contenido de las URLs relevantes encontradas por Serper.

    Tu resultado esperado es un **reporte conciso en texto** que liste las 3-5 tendencias o sub-tópicos
    más relevantes y activos que encontraste, mencionando brevemente por qué son relevantes y
    en qué plataforma (si se especifica en la fuente) parecen tener más fuerza.
    """,
    expected_output="Reporte de tendencias actuales en finanzas personales en redes sociales colombianas (texto plano).",
    agent=investigador_tendencias
)

# Tarea para el Agente: Analista de Contexto Local
# Objetivo: Identificar fechas relevantes para finanzas personales en Colombia.
# Dependencias: Ninguna inicial. Usa Tavily y process_search_tool.
# Nota: Esta tarea se podría ejecutar en paralelo o después de la primera, ya que su input no depende de la primera.
task_analizar_contexto = Task(
    description=f"""
    Identifica las fechas, festividades o eventos importantes que sean relevantes para las finanzas
    personales en COLOMBIA en la **fecha actual y las próximas dos semanas**.
    Considera eventos como plazos de declaración de renta (si aplica para la fecha), fechas de pago
    de primas (si aplica), festividades nacionales que puedan implicar gastos (vacaciones, navidad, etc.),
    o cualquier otro evento cultural o fiscal relevante para el manejo del dinero en Colombia.

    Utiliza las herramientas de búsqueda web para encontrar calendarios fiscales, noticias sobre fechas
    financieras clave o información sobre festividades colombianas y su relación con las finanzas.
    Utiliza 'process_search_tool' para leer los detalles de las URLs relevantes.

    Tu resultado esperado es una **lista en texto** de estas fechas o eventos próximos y una breve
    explicación de por qué son relevantes para el contenido de finanzas personales (ej: "Fecha límite
    declaración de renta - recordar documentos, cómo declarar", "Semana antes de navidad - consejos para
    no gastar de más").
    """,
    expected_output="Lista de fechas clave colombianas relevantes para finanzas personales (texto plano).",
    agent=analista_contexto
)


# Tarea para el Agente: Estratega de Contenidos Multiplataforma
# Objetivo: Crear ideas de contenido específicas combinando tendencias y contexto local.
# Dependencias: Requiere el output del Investigador de Tendencias y del Analista de Contexto.
task_estrategia_contenidos = Task(
    description=f"""
    Recibe el **reporte de tendencias** de redes sociales y la **lista de fechas relevantes** en Colombia.

    Tu tarea es generar de 7 a 10 **ideas de contenido concretas** para la aplicación PliP,
    basadas en la combinación de estas tendencias y el contexto local.
    Para cada idea de contenido, especifica:
    1.  **Tópico Principal:** (Basado en tendencias y/o fechas)
    2.  **Plataforma:** (Instagram, Facebook o TikTok - elige la más adecuada según la tendencia o formato)
    3.  **Formato Sugerido:** (Ej: Reel corto, Carrusel de IG, Post de Facebook con imagen/video, Guía descargable, Video explicativo de TikTok)
    4.  **Hora Sugerida:** (Mañana, Tarde o Noche - basándote en tu conocimiento de engagement por plataforma y tópico)
    5.  **Concepto/Ángulo:** Una breve descripción de qué trataría el contenido (ej: "3 tips rápidos para ahorrar en vacaciones", "Guía paso a paso para separar tu prima", "Mito o Realidad: ¿es mejor invertir que ahorrar? - Formato TikTok viral").
    6.  **Relevancia (Opcional pero útil):** Conecta la idea con la tendencia o fecha específica (ej: "Relacionado con tendencia #AhorroInteligente" o "Ideal para semana antes de declaración de renta").

    Tu resultado esperado es una **lista estructurada de ideas de contenido** (en texto plano, usando guiones, números o formato similar) que cubra aproximadamente una semana de publicaciones, variando entre plataformas, formatos y horas.
    """,
    expected_output="Lista estructurada de ideas de contenido multi-plataforma para una semana (texto plano).",
    agent=estratega_contenidos
)

# Tarea para el Agente: Generador de Parrillas Inteligentes
# Objetivo: Formatear las ideas en una estructura de parrilla semanal (texto).
# Dependencias: Requiere el output del Estratega de Contenidos. No tiene herramientas.
task_generar_parrilla = Task(
    description=f"""
    Recibe la **lista estructurada de ideas de contenido** para una semana.

    Tu tarea es organizar esta lista en un formato de **parrilla semanal en texto plano** que sea
    claro y fácil de entender, similar a cómo se vería una tabla simple.

    Incluye las siguientes columnas en tu salida de texto:
    -   **Día Sugerido** (Ej: Lunes, Martes, etc. - distribuye las ideas lógicamente)
    -   **Red Social**
    -   **Hora Sugerida** (Mañana, Tarde, Noche)
    -   **Tópico/Concepto del Contenido** (La descripción breve de la idea)
    -   **Formato Sugerido**

    El resultado esperado es una **tabla en texto plano** que represente la parrilla de contenido para los próximos 7 días, basada en las ideas proporcionadas. No generes un archivo real (.xlsx), solo el texto con el formato de tabla.
    """,
    expected_output="Parrilla de contenido semanal en formato de tabla de texto plano.",
    agent=generador_parrillas
)


# %%
agents = [investigador_tendencias, analista_contexto, estratega_contenidos, generador_parrillas]
crew = Crew(
    agents=agents,
    tasks=[task_investigar_tendencias, task_analizar_contexto, task_estrategia_contenidos, task_generar_parrilla],
   
    verbose=False
)

result = crew.kickoff()

print(result)

# %%
online_researcher = Agent(
    role="Online Researcher",
    goal="Research the topic online",
    backstory="""Your primary role is to function as an intelligent online research assistant, adept at scouring 
    the internet for the latest and most relevant trending stories across various sectors like politics, technology, 
    health, culture, and global events. You possess the capability to access a wide range of online news sources, 
    blogs, and social media platforms to gather real-time information.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm_genmini_2_0
    
)


