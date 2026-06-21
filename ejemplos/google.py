# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-3.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Hola?"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Responding to Greetings**

I've processed the initial \"Hola?\" and decided I need only a Spanish greeting in response, perhaps with an offer of help. No search queries are necessary for this simple exchange.


"""),
                types.Part.from_text(text="""¡Hola! ¿Cómo puedo ayudarte hoy?"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""ok"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""¿Hay algo en particular que te gustaría hacer o sobre lo que te gustaría hablar?"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""¿Puedes buscar en la documentación de Google acerca del bidding level para el SDK de Google?

Necesito saber:
1. ¿Qué otros parámetros acepta, aparte del medium?
2. ¿Cómo se deben enviar en el SDK la palabra correcta?"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Analyzing the Query**

I'm now focused on breaking down the Spanish query. I've identified the core concepts. The user is asking about \"bidding\" within Google's SDK documentation, likely related to mobile ads. I'm aiming to pinpoint the relevant Google resources. \"Bidding\" and \"medium\" are key.


**Investigating Bidding Context**

I'm now diving into the implications of \"bidding level\" within Google's SDK ecosystem. I'm trying to determine if this term, or something similar, exists within AdMob, Ad Manager, Tag Manager, or Analytics. Currently, my search focuses on \"bidding level,\" \"medium,\" and the contextual \"what\".


**Clarifying \"Medium\" Parameters**

I'm digging deeper into this \"medium\" concept. It seems like the client, or their documentation, might be referring to the \"medium rectangle\" ad size. I've found mentions in both Google's Mobile Ads SDK context and Meta Audience Network's documentation. The standard banner is also in question. I am now searching to see where this \"medium\" or \"medium rectangle\" is being referenced!


**Clarifying Terminology**

Okay, I've got it. I'm focusing in on the terminology, specifically relating to bidding levels. I was initially unsure, but I've narrowed it down to \"bid floor,\" then realized that in AdMob, these are configured as \"eCPM floors,\" with options like \"High floor,\" \"Medium floor,\" and \"Google optimized.\" And, in AdMob, there's a Google optimized floor.


**Investigating \"Bidding\" Parameters**

I've just noticed an \"eCPM floor\" setting in AdMob. It seems there are levels for \"Google Optimized\" floors, including \"High floor (Beta)\", \"Medium floor (Beta)\", and \"All prices\". Now, I'm going to investigate whether \"bidding\" is used in this context. I'll search for any related parameters.


**Re-analyzing the Queries**

Okay, I'm re-examining the Spanish query regarding Google's SDK bidding level, focusing on clarifying \"bidding level\" context. Also, I'm delving into identifying the other parameters the SDK might accept, besides \"medium\". I'm also working to determine the correct question format.


**Clarifying the AdMob Floors**

I've made a crucial link: the \"bidding level\" relates to AdMob's floor types: \"Google optimized,\" referring to \"high floor,\" \"medium floor,\" and \"all prices.\" I'm now verifying how these floors are configured. The AdMob UI seems to be the place to set floors at the Ad Unit level. I'm investigating if any parameters in `GADRequest` or `AdRequest`, or any Google SDK, allow passing these bidding levels. No luck with a `max_ad_content_rating` equivalent, so far.


**Investigating SDK Bidding**

I'm now diving into the different possible interpretations of \"bidding level\" in the context of the Google SDK. Could they be referring to Google Ads API bidding schemes like `MANUAL_CPC` or `TARGET_CPA`? Or is it something related to Google Tag Manager or Google Analytics? Now, I am considering the Spanish translation, and its context, particularly the role of the \"medium\" parameter. Perhaps it is about high, medium, and all floors?


**Clarifying Bidding Level**

I'm currently focused on the interpretation of \"bidding level.\" It appears to relate to eCPM floor settings within Google AdMob. I'm also exploring Google Ad Manager as a possible context. I'm investigating whether there may be other Google SDK possibilities.


**Investigating SDK Possibilities**

I'm now diving deeper into possible SDK solutions, specifically within Google's ecosystem. I've ruled out the User Consent SDK, and the concept of \"bidding level\" doesn't seem to apply. I'm looking beyond the UMP SDK and focusing on the relationship between RTB and potential API parameters like `BidRequest.imp.ext.ad_unit_mapping`. I'm also now investigating if anything relating to \"bidding level\" applies to Chrome's Privacy Sandbox, and if so, how. I was also curious about Analytics, or Tag Manager.


**Analyzing Campaign Parameters**

I'm now focused on clarifying potential mistranslations or typos. The user's reference to \"bidding level\" is throwing me. I'm considering it might be a misunderstanding of \"campaign level.\" I'm reviewing UTM parameters in GA4 and Firebase, to ascertain which ones the user is focused on. This includes `source`, `medium`, `campaign`, `term`, and `content`. I'm specifically looking at how these parameters are represented in the respective SDKs.


**Investigating Google AdMob**

Okay, I'm digging into Google's AdMob SDK documentation specifically on bidding levels. I'm focusing on identifying all acceptable parameters beyond \"medium\". Currently, I'm trying to decipher how to correctly format and submit the relevant keywords within the SDK environment.


**Exploring Safety Settings**

I've shifted focus to the Gemini AI SDK, specifically `SafetySetting`. I've confirmed that `HarmBlockThreshold` allows me to set levels beyond \"medium\". Options include `BLOCK_NONE`, `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, and `BLOCK_ONLY_HIGH`, and \"medium\" is `BLOCK_MEDIUM_AND_ABOVE`. I've excluded `thinkingBudget` and `bidding` as irrelevant. Now I am using the search string \"bidding level\" to explore related concepts in the Google Ads API.


**Analyzing AdMob Settings**

I'm now diving into the Google AdMob eCPM floor options: \"High floor\", \"Medium floor\", and \"All prices.\" I'm realizing these are the \"Google optimized\" levels, or bidding options, in the AdMob UI. Now, I'm beginning to wonder how they might be affecting the mediation configuration, and what impact that will have.


**Investigating eCPM Parameters**

I'm now clarifying the original query. It appears the question isn't about client-side SDK parameters for eCPM floors. I've confirmed that the eCPM floor is exclusively managed server-side within the AdMob platform. My investigation into the Google Ads API and Gemini API didn't reveal a \"bidding level\" parameter either. The query is now being reframed by looking at the phrasing.


**Clarifying Parameter Associations**

I'm now investigating \"medium\" within the context of SDK parameters. It seems \"medium\" is a standard Firebase Analytics campaign parameter (`FirebaseAnalytics.Param.MEDIUM`), alongside source, campaign, term, and content. This leads me to consider whether the \"bidding level\" terminology is a misinterpretation of campaign tracking or AdMob's eCPM floor settings. I'm exploring both interpretations.


**Defining AdMob Bidding Levels**

I'm now focusing on clarifying the \"bidding levels\" terminology. I've confirmed that the Google-optimized eCPM floors in AdMob, such as \"High floor,\" \"Medium floor,\" and \"All prices,\" are not configurable via the SDK. These options are configured within the AdMob console under the Ad Unit's \"Advanced Settings.\" I'm reviewing documentation, to be sure I am understanding all the options.


**Examining Campaign Parameters**

I'm now investigating Firebase Analytics parameters further. I've reconfirmed that Firebase Analytics uses predefined constants for parameters like `medium`, `source`, `campaign`, `term`, and `content`. I'm refreshing how these are configured within the Android and iOS SDKs. I'm also confirming that these campaign parameters have no relation to \"bidding levels\" or the AdMob server-side configurations. Finally, I'm reviewing the bidding strategy options within Google Ads campaigns.


**Differentiating Bidding Contexts**

I'm now clarifying the distinct contexts around \"bidding level\" and \"medium.\" I'm organizing my response to explain how these terms relate to AdMob (eCPM floors), Google Ads (bidding strategies like `MAXIMIZE_CONVERSIONS` or `MANUAL_CPC`), and Analytics (campaign parameters). I'm highlighting that \"bidding level\" and \"medium\" can be interpreted differently depending on the Google SDK in use. Also, if \"medium\" referred to ad sizes, then it would indeed relate to the Medium Rectangle size.


**Defining Ad Sizes**

I'm now identifying the ad sizes and formats within the Google Mobile Ads SDK. I'm focusing on clarifying how they're defined and passed within the Android, iOS, and Flutter SDKs, including their corresponding constants like `MEDIUM_RECTANGLE`. I'll also be explaining the deprecation of `SMART_BANNER` in favor of adaptive banners.


**Synthesizing Answers Concisely**

I'm now consolidating the separate answers. I'm articulating that \"All prices,\" \"high,\" and \"medium\" describe eCPM floor configurations in the AdMob console and can't be set in code. I'm differentiating this from the \"Medium Rectangle\" ad size, a specific banner format available in the mobile SDKs. Finally, I'm noting that \"medium\" also represents a campaign medium parameter in Firebase Analytics and is totally unrelated to bidding level. I'm now making sure I am not going around in circles.


"""),
                types.Part.from_text(text="""En la documentación de los SDK de Google, el concepto de **\"bidding level\"** (nivel de puja) o el parámetro **\"medium\"** pueden interpretarse en tres contextos de desarrollo distintos. 

Para asegurarnos de cubrir exactamente lo que necesitas, a continuación te presentamos las respuestas según el caso que estés implementando:

---

### Escenario 1: Si te refieres a los \"eCPM Floors\" de Google AdMob (Límites de Puja Optimizados por Google)
En AdMob, Google ofrece la funcionalidad de establecer límites de eCPM (pisos de puja) optimizados dinámicamente para proteger el valor de tu inventario frente a los compradores de anuncios.

1. **¿Qué otros parámetros/niveles acepta, aparte de medium?**
   Google ofrece tres niveles de optimización automática para los pisos de eCPM:
   * **High floor (Límite alto):** Prioriza mostrar solo anuncios de alto valor (maximiza el eCPM pero puede bajar tu tasa de cobertura/fill rate).
   * **Medium floor (Límite medio):** Busca un equilibrio constante entre ingresos por anuncio y tasa de cobertura.
   * **All prices (Todos los precios):** Maximiza la tasa de cobertura aceptando cualquier oferta disponible.

2. **¿Cómo se deben enviar en el SDK?**
   **No se pueden enviar mediante el código del SDK.** Google AdMob no permite que configures o alteres dinámicamente los pisos de eCPM optimizados desde la aplicación (lado del cliente) por razones de seguridad y estabilidad de la subasta. 
   * **Cómo se configuran:** Debes entrar a la **Consola web de AdMob**, ir a la configuración de tu *Ad Unit* (Bloque de anuncios), desplegar la *Configuración avanzada* y seleccionar el nivel deseado (Alto, Medio o Todos los precios). El SDK aplicará esta regla automáticamente en los servidores de Google.

---

### Escenario 2: Si te refieres al tamaño de anuncio \"Medium Rectangle\" (MREC) en el SDK de anuncios móviles (Google Mobile Ads SDK)
Si estás integrando anuncios y por \"medium\" te refieres al formato de rectángulo mediano (MREC, de 300x250 dp):

1. **¿Qué otros tamaños estándar acepta?**
   * **BANNER:** Banner estándar (320x50).
   * **LARGE_BANNER:** Banner grande (320x100).
   * **FULL_BANNER:** Banner completo para tabletas (468x60).
   * **LEADERBOARD:** Banner apaisado para tabletas (728x90).
   * Además de tamaños adaptativos mediante métodos como `AnchoredAdaptiveBanner` o `InlineAdaptiveBanner`.

2. **¿Cómo se deben enviar en el SDK con la palabra correcta?**
   Dependiendo de la plataforma de desarrollo que utilices, el SDK de Google espera una constante predefinida:
   * **Android (Java/Kotlin):** `AdSize.MEDIUM_RECTANGLE`
   * **iOS (Swift):** `GADAdSizeMediumRectangle`
   * **Flutter:** `AdSize.mediumRectangle`

---

### Escenario 3: Si te refieres a parámetros de campaña (medium / utm_medium) en Firebase/Google Analytics SDK
Si estás haciendo seguimiento de atribución o campañas de marketing y deseas registrar la procedencia del usuario:

1. **¿Qué otros parámetros acepta, aparte de medium?**
   Acepta los parámetros estándar de atribución UTM:
   * `source` (origen de la campaña, ej: \"google\")
   * `campaign` (nombre de la campaña)
   * `term` (palabras clave de búsqueda de pago)
   * `content` (contenido específico del anuncio)

2. **¿Cómo se deben enviar en el SDK con la palabra correcta?**
   Para enviar de forma correcta estos datos, el SDK de Google Analytics (Firebase) cuenta con constantes reservadas que se pasan dentro del evento de analíticas:
   * **Android (Kotlin/Java):** Usa el parámetro `FirebaseAnalytics.Param.MEDIUM`.
     ```kotlin
     val params = Bundle().apply {
         putString(FirebaseAnalytics.Param.MEDIUM, \"cpc\")
         putString(FirebaseAnalytics.Param.SOURCE, \"google\")
     }
     firebaseAnalytics.logEvent(FirebaseAnalytics.Event.CAMPAIGN_DETAILS, params)
     ```
   * **iOS (Swift):** Usa `AnalyticsParameterMedium`.
     ```swift
     Analytics.logEvent(AnalyticsEventCampaignDetails, parameters: [
         AnalyticsParameterMedium: \"cpc\",
         AnalyticsParameterSource: \"google\"
     ])
     ```

Si tu consulta está orientada a otro SDK específico de Google (como la API de Google Ads para agencias o anunciantes), no dudes en comentarlo para profundizar en los objetos correspondientes."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Cuando es el proximo partido de la seleccion colombia"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Pinpointing the Query**

I've successfully grasped the user's Spanish query: they want the date of the next Colombia national team match. I've also established that today's date is June 20, 2026. This is the foundation I need to find the specific match date.


**Anticipating the Next Match**

Okay, I'm focusing on Colombia's upcoming World Cup match. It's June 20, 2026. My understanding is that Colombia just had their debut match on June 17, 2026. They beat Uzbekistan 3-0. Now, I'm trying to figure out who's next for them in the tournament.


**Pinpointing the Next Game**

I've got the details of Colombia's next World Cup match locked down. It's against the Democratic Republic of Congo on Tuesday, June 23rd, at 9:00 PM Colombian time (20:00 local time) at Estadio Akron in Guadalajara, Mexico. I'm also confirming the subsequent match against Portugal on Saturday, June 27th, likely at 6:30 PM (or 7:30 PM) Colombian time in Miami.


"""),
                types.Part.from_text(text="""El próximo partido de la Selección Colombia masculina de mayores en la Copa Mundial de la FIFA 2026 es el siguiente:

* **Rival:** República Democrática del Congo (RD Congo)
* **Fecha:** Martes 23 de junio de 2026
* **Hora:** 9:00 p.m. (hora colombiana) / *nota: algunas programaciones oficiales de la FCF la listan a las 8:00 p.m.*
* **Lugar:** Estadio Guadalajara (Estadio Akron) en Guadalajara, México
* **Instancia:** Segunda fecha de la fase de grupos (Grupo K)

**Contexto del grupo:**
* Colombia llega a este compromiso como líder de su zona tras haber ganado 3-1 en su debut frente a Uzbekistán el pasado miércoles 17 de junio. 
* Su rival, la República Democrática del Congo, viene de conseguir un empate 1-1 ante Portugal.

**Siguiente partido:**
El tercer y último partido de la fase de grupos para la selección colombiana será contra **Portugal** el sábado 27 de junio de 2026, a las 6:30 p.m. (hora colombiana), en el Hard Rock Stadium de Miami, Estados Unidos."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch(
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MEDIUM",
        ),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",  # Block none
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",  # Block none
            ),
        ],
        tools=tools,
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if text := chunk.text:
            print(text, end="")

if __name__ == "__main__":
    generate()


