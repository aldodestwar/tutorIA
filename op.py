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
# User wants to display this name, but the actual API name is different
DISPLAY_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Display the actual model name now for clarity
ACTUAL_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Use the correct, standard identifier
# Increase context limit significantly for Gemini 1.5 Flash (1M tokens ~ 4M chars theoretical max)
# Setting a practical limit, e.g., 2 million chars, still very large.
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
         # Clearer error if the folder itself is missing
         error_msg = f"❌ **Error Crítico:** La carpeta '{data_dir}' no existe en la ubicación del script (`{script_dir}`). Por favor, créala y coloca tus archivos .txt dentro."
         st.error(error_msg) # Display error immediately
         return "", [], False, error_msg # Return error message

    if not files:
        warn_msg = f"⚠️ No se encontraron archivos '{file_pattern}' en la carpeta '{data_dir}'. Asegúrate de que el material de estudio (.txt) esté presente allí."
        # No need to return error here, sidebar will show this message
        return "", [], False, warn_msg

    # Use st.status for a collapsible loading animation
    with st.status("✨ Cargando base de conocimiento...", expanded=True) as status:
        total_chars = 0
        errors = []

        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                content = ""
                # Try reading with UTF-8 first, then latin-1 as fallback
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    status.update(label=f"⏳ Cargando base de conocimiento... (Intentando latin-1 para {file_name})")
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()

                file_names.append(file_name)
                # Add clear markers for the AI
                source_marker = f"\n\n--- INICIO DOCUMENTO: {file_name} ---\n\n"
                all_text += source_marker + content
                char_count = len(source_marker) + len(content)
                total_chars += char_count
                # Update status with current file progress
                status.write(f"📄 Cargado: **{file_name}** ({char_count:,} caracteres)")
                time.sleep(0.05) # Small delay for visual feedback

            except Exception as e:
                error_str = f"❌ Error procesando archivo {file_name}: {e}"
                errors.append(error_str)
                status.write(error_str) # Show error in status

        # Check if context exceeds the warning threshold
        context_is_large_warning = total_chars > MAX_CONTEXT_CHARS_WARN
        status_message = f"✅ Base de conocimiento cargada ({len(file_names)} archivos). Total ~{total_chars:,} caracteres."

        if context_is_large_warning:
             status_message += "\n\n⚠️ **Advertencia:** El contenido total es muy grande. Las respuestas pueden ser más lentas o costosas."

        if errors:
             status_message += f"\n\n❌ Se encontraron {len(errors)} errores al cargar algunos archivos."
             status.update(label="⚠️ Base de conocimiento cargada con errores.", state="warning", expanded=True)
        else:
             status.update(label="✅ Base de conocimiento cargada exitosamente.", state="complete", expanded=False) # Collapse on success

        # Return combined text, list of files, the warning flag, and any loading messages/errors
        return all_text, file_names, context_is_large_warning, status_message # Return status_message too


def get_gemini_response_full_context(api_key, full_context, user_query):
    """Generates response using Gemini 1.5 Flash with full context and structured output request."""
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error configurando Google AI: {e}")
        return "⚠️ Hubo un problema con la configuración de la IA. Verifica tu API Key."

    # --- Enhanced Prompt for Structured Output ---
    prompt = f"""**Instrucciones para el Chatbot TutorIA (Experto en Derecho Laboral Chileno):**

Eres un profesor experto en Derecho Laboral y Procedimientos de Chile, enfocado en enseñar a estudiantes de Técnico Jurídico. Tu objetivo es educar de forma clara, completa y estructurada.

**Contexto:** Te proporciono a continuación la **totalidad** del material de estudio disponible. El material está dividido por documentos, marcados con `--- INICIO DOCUMENTO: [nombre_archivo] ---`.

**Tarea:** Responde a la *última pregunta del estudiante* basándote **exclusivamente** en la información contenida dentro de **todo** este material de estudio.

**Formato de Respuesta (¡MUY IMPORTANTE!):**
*   **Estructura de Apunte:** Organiza tu respuesta como si fueran apuntes de clase claros y ordenados.
*   **Markdown:** Utiliza formato Markdown para mejorar la legibilidad:
    *   Usa encabezados (`## Título Principal`, `### Subtítulo`) para separar secciones lógicas.
    *   Utiliza listas con viñetas (`* Punto 1`, `* Punto 2`) o numeradas (`1. Paso 1`, `2. Paso 2`) para enumeraciones o pasos.
    *   Resalta **términos clave**, conceptos importantes o artículos legales usando **negrita**.
    *   Si citas textualmente una parte muy breve y relevante, puedes usar `> Cita textual...`
*   **Claridad y Detalle:** Sé **exhaustivo y detallado**. Explica los conceptos clave con terminología legal precisa pero **asegúrate de que sea comprensible** para un estudiante técnico. Si aplica, usa ejemplos simples para ilustrar.
*   **Citación de Fuentes:** ¡Fundamental! Después de explicar un concepto o responder una parte de la pregunta, **indica claramente el documento fuente** de donde obtuviste la información. Hazlo entre paréntesis al final del párrafo o sección relevante. Ejemplo: `(Fuente: NombreDelArchivo.txt)`. Si la información proviene de múltiples fuentes, cítalas todas: `(Fuentes: Archivo1.txt, Archivo2.txt)`.
*   **Tono:** Mantén un tono **profesoral, respetuoso y alentador**. Eres un guía educativo.
*   **Información No Encontrada:** Si después de revisar **todo** el material, la información específica para responder *no* está presente, indícalo amablemente: "Estimado/a estudiante, he revisado detenidamente todo el material de estudio (`[Lista de archivos revisados]`) y no encuentro información específica sobre [tema de la pregunta]. Es posible que este detalle no esté cubierto en los documentos proporcionados. ¿Puedo ayudarte con otro tema del material?"
*   **No Inventar:** No agregues información que no esté en el material proporcionado ni uses conocimiento externo.

**MATERIAL DE ESTUDIO COMPLETO:**
{full_context}
--- FIN MATERIAL DE ESTUDIO ---

**ÚLTIMA PREGUNTA DEL ESTUDIANTE:**
{user_query}

**TU RESPUESTA (Estructurada, detallada, basada en el material, citando fuentes):**
"""

    try:
        model = genai.GenerativeModel(ACTUAL_MODEL_NAME)
        # Standard safety settings
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
                temperature=0.4, # Slightly higher temp might encourage more "natural" structure
                # Consider adding max_output_tokens if needed, but 1.5 Flash has a large output limit
                )
        )

        # Handle response (same logic as before, but expecting Markdown now)
        if not response.candidates:
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Desconocida"
             safety_feedback = response.prompt_feedback.safety_ratings if response.prompt_feedback else []
             feedback_details = "; ".join([f"{fb.category.name}: {fb.probability.name}" for fb in safety_feedback])
             st.error(f"🔒 Respuesta bloqueada por seguridad ({block_reason}). Detalles: {feedback_details}")
             return f"⚠️ Estimado/a estudiante, mi respuesta fue bloqueada por razones de seguridad ({block_reason}). Por favor, asegúrate de que tu pregunta sea apropiada y esté relacionada directamente con el material de estudio de Derecho Laboral."

        # Check finish reason and extract text
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"

        if candidate.content and candidate.content.parts:
            generated_text = candidate.content.parts[0].text.strip()
            if finish_reason != "STOP" and finish_reason != "MAX_TOKENS":
                # Warn if stopped for safety or other reasons, but still show text
                st.warning(f"⚠️ La generación de la respuesta se detuvo por: **{finish_reason}**. La respuesta podría estar incompleta.")
                return generated_text + f"\n\n*(Respuesta posiblemente incompleta debido a: {finish_reason})*"
            elif not generated_text:
                 return "⚠️ Ocurrió un problema técnico: la IA generó una respuesta vacía."
            else:
                 return generated_text # Successful generation
        else:
             # This case means no content was generated at all
             return f"⚠️ Ocurrió un problema técnico al generar la respuesta (Razón de finalización: {finish_reason}, sin contenido)."


    except Exception as e:
        # Refined error handling messages
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
             # Check for safety block within InvalidArgumentError (sometimes happens)
             if "safety" in error_str:
                  return "⚠️ **Respuesta Bloqueada:** La solicitud o la respuesta potencial fue bloqueada por las políticas de seguridad. Asegúrate que la pregunta sea apropiada."
             else:
                  return "⚠️ **Error: Solicitud Inválida.** Hubo un problema con los datos enviados a la IA. Intenta reformular tu pregunta."
        else:
            st.error(f"❌ Error inesperado al generar respuesta con Gemini API: {e}")
            return "⚠️ Lo siento, ocurrió un error inesperado al procesar tu solicitud. Por favor, inténtalo de nuevo."

# --- Streamlit App UI ---
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
            # Basic check: Configure genai to see if the key format seems valid enough
            try:
                 genai.configure(api_key=st.session_state.google_api_key)
                 # Test connection (optional but good) - maybe make a small dummy call? For now, configure is enough.
                 st.sidebar.success("API Key aceptada. ✅")
                 time.sleep(1) # Brief pause
            except Exception as e:
                 st.sidebar.error(f"Error configurando Google AI: {e}. Verifica la clave. ❌")
                 st.session_state.google_api_key = None
                 st.session_state.api_key_confirmed = False
            st.rerun()
        else:
            st.sidebar.warning("🚨 Por favor, ingresa una clave API.")
else:
    # Display partial key for confirmation without exposing it fully
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
    # load_full_text_from_data now returns the status message too
    full_text, loaded_f, is_large, load_status_message = load_full_text_from_data(
        DATA_FOLDER, CONTEXT_FILE_PATTERN
    )
    # Display the final load status message in the sidebar
    if "Error Crítico" in load_status_message:
        # Error already shown by the function
        pass
    elif "No se encontraron archivos" in load_status_message:
         st.sidebar.warning(load_status_message) # Show warning if no files
    elif loaded_f: # Check if any files were successfully loaded
        st.session_state.full_text_content = full_text
        st.session_state.loaded_files = loaded_f
        st.session_state.context_is_large_warning = is_large
        st.session_state.full_context_ready = True
        # Display success/warning message from loading function
        if "⚠️" in load_status_message:
            st.sidebar.warning(load_status_message)
        else:
            st.sidebar.info(load_status_message)

        # Add initial message only if context is ready and messages are empty
        if not st.session_state.messages:
             initial_greeting = f"¡Hola! 👋 Soy tu TutorIA de Derecho Laboral. He cargado {len(st.session_state.loaded_files)} documento(s) de estudio. ¿En qué te puedo ayudar hoy?"
             if st.session_state.context_is_large_warning:
                  initial_greeting += "\n\n*(⚠️ Advertencia: El material es extenso, las respuestas podrían tardar un poco.)*"
             st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
             st.rerun() # Rerun to show initial message and file list

    # If load_full_text_from_data returned an empty message (shouldn't happen with new logic, but safe)
    elif not loaded_f and not ("Error" in load_status_message or "No se encontraron" in load_status_message):
         st.sidebar.error("Ocurrió un error inesperado al cargar los archivos.")


# Display loaded files status in sidebar if ready
if st.session_state.full_context_ready and st.session_state.loaded_files:
    st.sidebar.subheader("📄 Archivos Cargados")
    # Use an expander for the file list if it's long
    with st.sidebar.expander(f"Ver {len(st.session_state.loaded_files)} archivos cargados", expanded=False):
        for file_name in st.session_state.loaded_files:
            # Shorten long names for display
            display_name = textwrap.shorten(file_name, width=35, placeholder="...")
            st.markdown(f"- `{display_name}`")

    if st.session_state.context_is_large_warning:
        st.sidebar.warning("⚠️ El contexto total es muy grande. Podría afectar el rendimiento.")
    else:
         st.sidebar.success("✅ Material listo para consulta.")


# --- Main Chat Area ---
st.divider() # Separator before chat history

if not st.session_state.google_api_key:
     st.info("👈 Ingresa tu Google API Key en la barra lateral para comenzar a chatear. ✨")
elif not st.session_state.full_context_ready:
     # Loading message is now handled by the sidebar status/info boxes
     st.info("⏳ Esperando la carga del material de estudio desde la carpeta 'data'... Revisa la barra lateral para ver el progreso. ✨")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Use columns for structure, maybe add avatar later
            # col1, col2 = st.columns([1,15]) # Example: small column for icon
            # with col1:
            #     st.write("🧑‍🏫" if message["role"] == "assistant" else "🧑‍🎓")
            # with col2:
            st.markdown(message["content"], unsafe_allow_html=False) # Render markdown

    # Chat input field
    if prompt := st.chat_input("Escribe tu pregunta aquí sobre el material del curso... 🤔"):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            # col1, col2 = st.columns([1,15])
            # with col1: st.write("🧑‍🎓")
            # with col2:
            st.markdown(prompt)

        # Generate and display AI response
        with st.chat_message("assistant"):
            # col1, col2 = st.columns([1,15])
            # with col1: st.write("🧑‍🏫")
            # with col2:
             # Use st.spinner for a clear processing indicator during generation
             with st.spinner("✨ Pensando y buscando en el material..."):
                 full_response = get_gemini_response_full_context(
                     st.session_state.google_api_key,
                     st.session_state.full_text_content,
                     prompt
                 )
             # Display the potentially Markdown-formatted response
             st.markdown(full_response, unsafe_allow_html=False) # Ensure HTML is not allowed by default

        # Add AI response to state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun() # Rerun to ensure the input box is cleared and layout is updated correctly


    # Button to clear chat history (place it less prominently, maybe sidebar or end of chat?)
    if len(st.session_state.messages) > 1: # Show only if there's history beyond the initial greeting
        if st.button("🧹 Limpiar Conversación"):
            # Keep only the initial greeting if it exists
            initial_message = next((m for m in st.session_state.messages if m['role'] == 'assistant'), None)
            st.session_state.messages = [initial_message] if initial_message else []
            st.rerun()


# --- Footer/Notes in Sidebar ---
st.sidebar.divider()
st.sidebar.caption("📝 Notas Técnicas:")
st.sidebar.caption(f"IA: `{ACTUAL_MODEL_NAME}`")
st.sidebar.caption("Modo: Contexto Completo (Todo el texto de 'data' se envía a la IA).")
st.sidebar.caption("Puede ser lento/costoso si los archivos .txt son muy grandes.")
st.sidebar.caption("Requiere Google API Key válida.")
st.sidebar.markdown("---")
st.sidebar.caption("✨ App por Aldo Manuel Herrera Hernández - IPP")