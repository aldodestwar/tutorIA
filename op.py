import streamlit as st
import google.generativeai as genai
import os
import glob
import time
import textwrap # Import textwrap for potentially shortening large text for display

# --- Configuration ---
APP_TITLE = "📚 TutorIA Derecho Laboral - Aldo Manuel Herrera Hernández - IPP"
DATA_FOLDER = "data"
CONTEXT_FILE_PATTERN = "*.txt"
DISPLAY_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Display the actual model name now for clarity
ACTUAL_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Use the correct, standard identifier
MAX_CONTEXT_CHARS_WARN = 2000000 # Warn if context exceeds this, but don't block unless API fails
API_KEY_LINK = "https://aistudio.google.com/apikey"

# --- Initialize session state ---
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = None
if 'api_key_confirmed' not in st.session_state:
     st.session_state.api_key_confirmed = False
if 'full_context_ready' not in st.session_state:
    st.session_state.full_context_ready = False
    st.session_state.full_text_content = ""
    st.session_state.loaded_files = []
    st.session_state.context_is_large_warning = False # Flag for large context warning
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---

@st.cache_data(show_spinner=False) # Cache the loading process
def load_full_text_from_data(data_dir, file_pattern):
    """Loads and concatenates text from all files in data_dir with st.status animation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(script_dir, data_dir)
    files = glob.glob(os.path.join(full_data_path, file_pattern))
    all_text = ""
    file_names = []

    if not os.path.exists(full_data_path):
         error_msg = f"❌ **Error Crítico:** La carpeta '{data_dir}' no existe en la ubicación del script (`{script_dir}`). Por favor, créala y coloca tus archivos .txt dentro."
         st.error(error_msg)
         return "", [], False, error_msg

    if not files:
        warn_msg = f"⚠️ No se encontraron archivos '{file_pattern}' en la carpeta '{data_dir}'. Asegúrate de que el material de estudio (.txt) esté presente allí."
        return "", [], False, warn_msg

    with st.status("✨ Cargando base de conocimiento...", expanded=True) as status:
        total_chars = 0
        errors = []

        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                content = ""
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    status.update(label=f"⏳ Cargando base de conocimiento... (Intentando latin-1 para {file_name})")
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()

                file_names.append(file_name)
                source_marker = f"\n\n--- INICIO DOCUMENTO: {file_name} ---\n\n"
                all_text += source_marker + content
                char_count = len(source_marker) + len(content)
                total_chars += char_count
                status.write(f"📄 Cargado: **{file_name}** ({char_count:,} caracteres)")
                time.sleep(0.05)

            except Exception as e:
                error_str = f"❌ Error procesando archivo {file_name}: {e}"
                errors.append(error_str)
                status.write(error_str)

        context_is_large_warning = total_chars > MAX_CONTEXT_CHARS_WARN
        status_message = f"✅ Base de conocimiento cargada ({len(file_names)} archivos). Total ~{total_chars:,} caracteres."

        if context_is_large_warning:
             status_message += "\n\n⚠️ **Advertencia:** El contenido total es muy grande. Las respuestas pueden ser más lentas o costosas."

        if errors:
             status_message += f"\n\n❌ Se encontraron {len(errors)} errores al cargar algunos archivos."
             status.update(label="⚠️ Base de conocimiento cargada con errores.", state="warning", expanded=True)
        else:
             status.update(label="✅ Base de conocimiento cargada exitosamente.", state="complete", expanded=False)

        return all_text, file_names, context_is_large_warning, status_message


def get_gemini_response_full_context(api_key, full_context, user_query):
    """Generates response using Gemini 1.5 Flash with full context, structured output, and strict constraints."""
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error configurando Google AI: {e}")
        return "⚠️ Hubo un problema con la configuración de la IA. Verifica tu API Key."

    # --- Enhanced Prompt with Strict Constraints ---
    prompt = f"""**Instrucciones para el Chatbot TutorIA (Experto en Derecho Laboral Chileno):**

Eres un profesor experto en Derecho Laboral y Procedimientos de Chile, enfocado exclusivamente en enseñar a estudiantes de Técnico Jurídico utilizando el material proporcionado. Tu objetivo es educar de forma clara, completa, estructurada y **estrictamente dentro de los límites del material y las reglas de comportamiento**.

**Contexto:** Te proporciono a continuación la **totalidad** del material de estudio disponible. El material está dividido por documentos, marcados con `--- INICIO DOCUMENTO: [nombre_archivo] ---`.

**Tarea Principal:** Responde a la *última pregunta del estudiante* basándote **única y exclusivamente** en la información contenida dentro de **todo** este material de estudio.

**Reglas Estrictas de Respuesta:**

1.  **Enfoque Exclusivo:**
    *   Tu única función es explicar conceptos y responder preguntas sobre **Derecho Laboral Chileno** tal como se presenta en el **material de estudio proporcionado**.
    *   **Si la pregunta del estudiante *NO* está relacionada directamente con el contenido del material de estudio (por ejemplo, preguntas sobre otros temas, historia general, opiniones personales, el clima, matemáticas, ciencia, etc.), DEBES RECHAZAR AMABLEMENTE la pregunta.** Indica claramente que tu propósito es ayudar *solo* con el material del curso de Derecho Laboral. Ejemplo de rechazo: "Mi función es ayudarte a entender el material de Derecho Laboral Chileno proporcionado. No tengo información sobre [tema no relacionado] y no puedo responder preguntas fuera de ese ámbito. ¿Tienes alguna consulta sobre el contenido del curso?"
    *   **No inventes información.** No agregues datos, ejemplos o explicaciones que no se deriven directamente del texto proporcionado. No utilices conocimiento externo.

2.  **Formato de Respuesta (¡Obligatorio!):**
    *   **Estructura de Apunte:** Organiza tu respuesta como apuntes de clase claros y ordenados.
    *   **Markdown:** Utiliza formato Markdown para mejorar la legibilidad: Encabezados (`##`, `###`), listas (`*`, `1.`), **negrita** para términos clave/artículos, y citas breves (`> Cita...`) si es muy relevante.
    *   **Claridad y Detalle:** Sé exhaustivo y detallado dentro de lo que permite el material. Usa terminología legal precisa pero explícala de forma comprensible para un estudiante técnico.
    *   **Citación de Fuentes:** ¡Fundamental! Después de cada bloque de información, **indica el documento fuente** entre paréntesis. Ejemplo: `(Fuente: NombreDelArchivo.txt)`. Si usas varias fuentes para una sección, cítalas todas: `(Fuentes: Archivo1.txt, Archivo2.txt)`.

3.  **Comportamiento y Tono:**
    *   Mantén siempre un tono **profesoral, estrictamente profesional, respetuoso, neutral y alentador**.
    *   **PROHIBIDO:** Usar lenguaje ofensivo, insultante, discriminatorio, sarcástico o irrespetuoso.
    *   **PROHIBIDO:** Emitir juicios de valor negativos sobre personas, grupos, entidades o situaciones. No critiques ni hables mal de nadie.
    *   **PROHIBIDO:** Participar en conversaciones inapropiadas o que violen las políticas de seguridad.

4.  **Información No Encontrada (Dentro del Contexto):**
    *   Si la pregunta *sí* es sobre Derecho Laboral Chileno, pero después de revisar **todo** el material, la información específica para responder *no* está presente, indícalo amablemente: "Estimado/a estudiante, he revisado detenidamente todo el material de estudio (`[Lista de archivos revisados]`) y no encuentro información específica sobre [tema de la pregunta dentro del derecho laboral]. Es posible que este detalle no esté cubierto en los documentos proporcionados. ¿Puedo ayudarte con otro tema del material?"

**MATERIAL DE ESTUDIO COMPLETO:**
{full_context}
--- FIN MATERIAL DE ESTUDIO ---

**ÚLTIMA PREGUNTA DEL ESTUDIANTE:**
{user_query}

**TU RESPUESTA (Siguiendo estrictamente TODAS las reglas anteriores):**
"""

    try:
        model = genai.GenerativeModel(ACTUAL_MODEL_NAME)
        # Standard safety settings - These act as a final backstop
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3, # Slightly lower temperature to encourage sticking to instructions
                # max_output_tokens=... # Consider setting if needed, but 1.5 Flash output is large
                )
        )

        # --- Response Handling ---
        if not response.candidates:
             # Handle blocked responses due to safety or other reasons
             block_reason = "Desconocida"
             feedback_details = "No disponible"
             if response.prompt_feedback:
                 block_reason = response.prompt_feedback.block_reason or "Desconocida"
                 if response.prompt_feedback.safety_ratings:
                      feedback_details = "; ".join([f"{fb.category.name}: {fb.probability.name}" for fb in response.prompt_feedback.safety_ratings])

             # More user-friendly message for safety blocks
             if block_reason == "SAFETY":
                  st.error(f"🔒 La pregunta o la respuesta potencial fue bloqueada por políticas de seguridad. Detalles: {feedback_details}")
                  return f"⚠️ Estimado/a estudiante, tu pregunta o la respuesta que iba a generar fue bloqueada por razones de seguridad. Por favor, asegúrate de que tu consulta sea apropiada, respetuosa y esté relacionada directamente con el material de estudio de Derecho Laboral."
             else:
                  st.error(f"🔒 Respuesta bloqueada por la API ({block_reason}). Detalles: {feedback_details}")
                  return f"⚠️ Estimado/a estudiante, hubo un problema al generar la respuesta (Bloqueo: {block_reason}). Por favor, intenta reformular tu pregunta o verifica que esté relacionada con el material."


        # Check finish reason and extract text
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"

        if candidate.content and candidate.content.parts:
            generated_text = candidate.content.parts[0].text.strip()

            # Check for potential refusals based on prompt instructions (heuristic check)
            refusal_keywords = ["no tengo información sobre", "mi función es ayudarte", "fuera de ese ámbito", "no puedo responder"]
            is_likely_refusal = any(keyword in generated_text.lower() for keyword in refusal_keywords)

            if finish_reason == "SAFETY":
                 # If it finished due to safety, even if text exists, prioritize safety message
                 st.warning(f"⚠️ La generación de la respuesta se detuvo por: **{finish_reason}**. Mostrando contenido parcial si existe, pero podría ser inapropiado.")
                 return f"⚠️ **Respuesta Bloqueada Parcialmente por Seguridad:**\n\n{generated_text}\n\n*(Advertencia: La respuesta completa fue bloqueada por seguridad. El contenido mostrado puede ser incompleto o problemático.)*"
            elif finish_reason not in ["STOP", "MAX_TOKENS"]:
                # Warn if stopped for other reasons (e.g., RECITATION, OTHER)
                st.warning(f"⚠️ La generación de la respuesta se detuvo por: **{finish_reason}**. La respuesta podría estar incompleta.")
                return generated_text + f"\n\n*(Respuesta posiblemente incompleta debido a: {finish_reason})*"
            elif not generated_text and not is_likely_refusal:
                 # Empty response, not a polite refusal -> Likely an error
                 return f"⚠️ Ocurrió un problema técnico: la IA generó una respuesta vacía (Razón: {finish_reason})."
            else:
                 # Successful generation or intentional refusal based on prompt
                 return generated_text
        else:
             # No content generated at all
             return f"⚠️ Ocurrió un problema técnico al generar la respuesta (Razón de finalización: {finish_reason}, sin contenido)."


    except Exception as e:
        # (Error handling remains the same as before - it's already quite detailed)
        error_str = str(e).lower()
        if "api_key" in error_str or "permission denied" in error_str:
             st.error(f"🔑 Error de API: Clave API inválida, sin permisos para el modelo '{ACTUAL_MODEL_NAME}', o problema de facturación. Verifica tu clave y cuenta de Google AI. ({e})")
             return f"⚠️ **Error de Autenticación/Permiso:** No se pudo acceder al modelo '{ACTUAL_MODEL_NAME}'. Por favor, verifica que tu API Key sea correcta, esté activa y tenga los permisos necesarios. Consulta el enlace en la barra lateral."
        elif "resource_exhausted" in error_str or "quota" in error_str:
             st.error(f"❌ Error de API: Cuota de uso excedida. ({e})")
             return "⚠️ **Error: Límite de Uso Alcanzado.** Has excedido la cuota permitida para la API de Google AI. Inténtalo más tarde o revisa tu plan."
        elif "deadline_exceeded" in error_str:
             st.error(f"⏳ Error de API: Tiempo de espera agotado. El contexto o la pregunta pueden ser demasiado complejos. ({e})")
             return "⚠️ **Error: Tiempo de Espera Excedido.** La solicitud tardó demasiado en procesarse. Esto puede ocurrir con material muy extenso o preguntas muy amplias. Intenta ser más específico/a."
        elif "model_name" in error_str or "not found" in error_str:
             st.error(f"🤖 Error de API: Modelo '{ACTUAL_MODEL_NAME}' no encontrado o inválido. ({e})")
             return f"⚠️ **Error: Modelo No Encontrado.** El modelo '{ACTUAL_MODEL_NAME}' no está disponible o el nombre es incorrecto."
        elif "invalid_argument" in error_str:
             st.error(f"🤔 Error de API: Argumento inválido. Revisa la pregunta o el contexto. ({e})")
             if "safety" in error_str:
                  return "⚠️ **Respuesta Bloqueada:** La solicitud o la respuesta potencial fue bloqueada por las políticas de seguridad. Asegúrate que la pregunta sea apropiada."
             else:
                  return "⚠️ **Error: Solicitud Inválida.** Hubo un problema con los datos enviados a la IA. Intenta reformular tu pregunta."
        else:
            st.error(f"❌ Error inesperado al generar respuesta con Gemini API: {e}")
            return "⚠️ Lo siento, ocurrió un error inesperado al procesar tu solicitud. Por favor, inténtalo de nuevo."


# --- Streamlit App UI (No changes needed here from the previous version) ---
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(f"🏛️ {APP_TITLE}") # Added emoji to title
st.caption(f"(Tú tutor artificial de Derecho Laboral - IPP)")
st.markdown(f"🤖 Hola! Soy tu TutorIA. Pregúntame sobre el material del curso. Intentaré explicarte como un profesor, usando **sólo** las fuentes que Aldo Herrera me instruyó.")
st.info(f"🧠 **Modelo IA:** `{DISPLAY_MODEL_NAME}` (Entrenado por Aldo Manuel Herrera)")


# --- Sidebar Setup ---
st.sidebar.header("⚙️ Configuración y Estado")
st.sidebar.divider() # Visual separator

# API Key Handling
st.sidebar.subheader("🔑 API Key de Google Gemini")
if not st.session_state.google_api_key:
    st.sidebar.markdown(f"Necesitas una API Key para usar la IA. Obtenla aquí:")
    st.sidebar.page_link(API_KEY_LINK, label="🔗 Obtener Google API Key", icon="🔑")
    entered_key = st.sidebar.text_input("Ingresa tu Google API Key:", type="password", key="api_key_input", help="Tu clave no se guardará permanentemente.")
    if st.sidebar.button("Confirmar API Key ✨", type="primary"):
        if entered_key:
            st.session_state.google_api_key = entered_key
            st.session_state.api_key_confirmed = True
            try:
                 genai.configure(api_key=st.session_state.google_api_key)
                 # Simple test: List models to check connectivity/auth
                 # models = [m for m in genai.list_models()] # This can be slow
                 # if not models: raise Exception("No models found")
                 st.sidebar.success("API Key aceptada y configurada. ✅")
                 time.sleep(1)
            except Exception as e:
                 st.sidebar.error(f"Error configurando/verificando Google AI: {e}. Verifica la clave. ❌")
                 st.session_state.google_api_key = None
                 st.session_state.api_key_confirmed = False
            st.rerun()
        else:
            st.sidebar.warning("🚨 Por favor, ingresa una clave API.")
else:
    masked_key = st.session_state.google_api_key[:4] + "****" + st.session_state.google_api_key[-4:]
    st.sidebar.success(f"API Key cargada ({masked_key}). ✅")
    if st.sidebar.button("🗑️ Cambiar/Borrar API Key"):
        keys_to_reset = ['google_api_key', 'api_key_confirmed', 'full_context_ready',
                         'full_text_content', 'loaded_files', 'messages', 'context_is_large_warning']
        for key in keys_to_reset:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

st.sidebar.divider()

# Load Full Text Context after API key confirmation
st.sidebar.subheader("📚 Base de Conocimiento")
if st.session_state.api_key_confirmed and not st.session_state.full_context_ready:
    full_text, loaded_f, is_large, load_status_message = load_full_text_from_data(
        DATA_FOLDER, CONTEXT_FILE_PATTERN
    )
    if "Error Crítico" in load_status_message:
        pass # Error shown in main area by function
    elif "No se encontraron archivos" in load_status_message:
         st.sidebar.warning(load_status_message)
    elif loaded_f:
        st.session_state.full_text_content = full_text
        st.session_state.loaded_files = loaded_f
        st.session_state.context_is_large_warning = is_large
        st.session_state.full_context_ready = True
        if "⚠️" in load_status_message:
            st.sidebar.warning(load_status_message)
        else:
            st.sidebar.info(load_status_message)

        if not st.session_state.messages:
             initial_greeting = f"¡Hola! 👋 Soy tu TutorIA de Derecho Laboral. He cargado {len(st.session_state.loaded_files)} documento(s) de estudio: `{', '.join(st.session_state.loaded_files)}`. Mi propósito es ayudarte a entender **únicamente** este material. ¿En qué te puedo ayudar hoy sobre estos temas?"
             if st.session_state.context_is_large_warning:
                  initial_greeting += "\n\n*(⚠️ Advertencia: El material es extenso, las respuestas podrían tardar un poco.)*"
             st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
             st.rerun()

    elif not loaded_f and not ("Error" in load_status_message or "No se encontraron" in load_status_message):
         st.sidebar.error("Ocurrió un error inesperado al cargar los archivos.")


# Display loaded files status in sidebar if ready
if st.session_state.full_context_ready and st.session_state.loaded_files:
    st.sidebar.subheader("📄 Archivos Cargados")
    with st.sidebar.expander(f"Ver {len(st.session_state.loaded_files)} archivos cargados", expanded=False):
        for file_name in st.session_state.loaded_files:
            display_name = textwrap.shorten(file_name, width=35, placeholder="...")
            st.markdown(f"- `{display_name}`")

    if st.session_state.context_is_large_warning:
        st.sidebar.warning("⚠️ El contexto total es muy grande. Podría afectar el rendimiento.")
    else:
         st.sidebar.success("✅ Material listo para consulta.")


# --- Main Chat Area ---
st.divider()

if not st.session_state.google_api_key:
     st.info("👈 Ingresa tu Google API Key en la barra lateral para comenzar a chatear. ✨")
elif not st.session_state.full_context_ready:
     st.info("⏳ Esperando la carga del material de estudio desde la carpeta 'data'... Revisa la barra lateral para ver el progreso. ✨")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=False)

    # Chat input field
    if prompt := st.chat_input("Escribe tu pregunta aquí (sólo sobre Derecho Laboral del material)... 🤔"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
             with st.spinner("✨ Pensando y buscando en el material..."):
                 full_response = get_gemini_response_full_context(
                     st.session_state.google_api_key,
                     st.session_state.full_text_content,
                     prompt
                 )
             # Display response (already handles markdown)
             st.markdown(full_response, unsafe_allow_html=False) # Ensure HTML is not allowed

        # Add AI response to state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Only rerun if input was processed successfully (avoid rerunning on API errors shown in response)
        if not full_response.startswith("⚠️"):
            st.rerun()
        else: # If it was an error message, don't rerun to keep the error visible before next input
             pass


    # Button to clear chat history
    if len(st.session_state.messages) > 1:
        # Place clear button less prominently at the bottom or sidebar
        st.markdown("---") # Add a separator before the clear button
        if st.button("🧹 Limpiar Conversación"):
            # Keep only the initial greeting
            initial_message = st.session_state.messages[0] if st.session_state.messages and st.session_state.messages[0]['role'] == 'assistant' else None
            st.session_state.messages = [initial_message] if initial_message else []
            st.rerun()


# --- Footer/Notes in Sidebar ---
st.sidebar.divider()
st.sidebar.caption("📝 Notas Técnicas:")
st.sidebar.caption(f"IA: `{ACTUAL_MODEL_NAME}`")
st.sidebar.caption("Modo: Contexto Completo (Todo el texto de 'data' se envía a la IA).")
st.sidebar.caption("Restricciones: Sólo responde sobre el material; Tono profesional; No ofensivo.")
st.sidebar.caption("Puede ser lento/costoso si los archivos .txt son muy grandes.")
st.sidebar.caption("Requiere Google API Key válida.")
st.sidebar.markdown("---")
st.sidebar.caption("✨ App por Aldo Manuel Herrera Hernández - IPP")