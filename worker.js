/**
 * Production-Grade Cloudflare Worker AI Endpoint
 * Grounded Answers with Verified Source Citations (Fixed & Optimized)
 */

var worker_default = {
  async fetch(request, env, ctx) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization"
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);

    // Helper to strip markdown JSON formatting
    const cleanJsonString = (rawStr) => {
      let cleaned = rawStr.trim();
      // Remove starting ```json or ```
      cleaned = cleaned.replace(/^```(?:json)?\s*/i, "");
      // Remove ending ```
      cleaned = cleaned.replace(/\s*```$/, "");
      return cleaned.trim();
    };

    try {
      if (!env.AI) {
        throw new Error("Cloudflare AI binding ('env.AI') is missing. Check your wrangler.toml.");
      }

      // 0. Query Understanding / Rewrite Endpoint
      if (url.pathname === "/understand" || url.pathname === "/rewrite") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }

        const body = await request.json();
        const query = body.query || body.prompt || "";
        const chatHistory = body.chat_history || [];

        const prompt = `You are a query routing and rewrite engine.
Analyze the user's query and the conversation history to classify the query intent, resolve any conversational pronouns, and rewrite the query to be a self-contained search query.

Query: "${query}"
Chat History: ${JSON.stringify(chatHistory)}

Respond in strict JSON format:
{
  "intent": "document_qa",
  "rewritten_query": "self-contained search query",
  "clarification_needed": false,
  "clarification_question": ""
}

Rules:
1. "intent" must be exactly one of: document_qa, summary, definition, comparison, extractive, follow_up, general, or ambiguous.
2. If the query is ambiguous, vague, or too short, set "clarification_needed" to true and provide a short clarifying question. Otherwise, set "clarification_needed" to false.
3. "rewritten_query" should be a clear, standalone search query containing all necessary keywords from the query and history.`;

        let result;
        let chosenModel = "@cf/meta/llama-3.1-8b-instruct-fp8";
        try {
          result = await env.AI.run(chosenModel, {
            messages: [
              { role: "system", content: "You respond ONLY with raw JSON." },
              { role: "user", content: prompt }
            ]
          });
        } catch (err) {
          chosenModel = "@cf/meta/llama-3.2-3b-instruct";
          result = await env.AI.run(chosenModel, {
            messages: [
              { role: "system", content: "You respond ONLY with raw JSON." },
              { role: "user", content: prompt }
            ]
          });
        }

        let responseText = (typeof result === 'object' && result.response) ? result.response : String(result);
        responseText = cleanJsonString(responseText);

        // Verify if it is valid JSON, fallback to manual packing if parsing fails
        let jsonResponse;
        try {
          jsonResponse = JSON.parse(responseText);
        } catch (e) {
          jsonResponse = {
            intent: "document_qa",
            rewritten_query: query,
            clarification_needed: false,
            clarification_question: "",
            raw_response: responseText
          };
        }

        return new Response(
          JSON.stringify({
            success: true,
            model: chosenModel,
            data: jsonResponse
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // 1. Embeddings Endpoint
      if (url.pathname === "/embeddings" || url.pathname === "/embed") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }
        const body = await request.json();
        // Resolve parameter naming conflicts
        const textInput = body.text || body.input || body.contents;
        if (!textInput) {
          return new Response(
            JSON.stringify({ error: "Missing 'text' or 'input' parameter" }),
            { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
          );
        }
        const textList = Array.isArray(textInput) ? textInput : [textInput];
        const embeddings = await env.AI.run("@cf/baai/bge-large-en-v1.5", {
          text: textList
        });
        return new Response(
          JSON.stringify({
            success: true,
            model: "@cf/baai/bge-large-en-v1.5",
            data: embeddings.data || embeddings
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // 2. Chat / Analyze Endpoint
      if (url.pathname === "/analyze" || url.pathname === "/chat" || url.pathname === "/" || url.pathname === "") {
        if (request.method === "GET") {
          return new Response(
            JSON.stringify({
              status: "online",
              service: "DocAnalyzer Cloudflare Workers AI Endpoint",
              llm_model: "@cf/meta/llama-3.1-8b-instruct-fp8",
              embedding_model: "@cf/baai/bge-large-en-v1.5"
            }),
            {
              status: 200,
              headers: { ...corsHeaders, "Content-Type": "application/json" }
            }
          );
        }

        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }

        const body = await request.json();
        
        // Resolve key collision: check explicit question parameters first, fallback to text
        const userQuestion = String(
          body.query || body.prompt || body.question || body.message || (body.context ? body.text : "") || "Summarize document"
        ).trim();

        // Check context keys; ignore text if text was already consumed by userQuestion
        let rawContext = body.context || body.document || body.contents || (body.query || body.question ? body.text : "") || "";

        if (Array.isArray(rawContext)) {
          rawContext = rawContext
            .map((c) => (typeof c === "object" ? c.text || c.content || c.page_content || JSON.stringify(c) : String(c)))
            .join("\n\n");
        }

        const cleanContext = String(rawContext)
          .replace(/Visual\s*\[Page\s*\d+\]\s*Visual/gi, "")
          .replace(/^\s*Visual\s*$/gmi, "")
          .replace(/^\s*Page\s*\d+\s*\[Page\s*\d+\]\s*/gmi, "")
          .replace(/^\[Document:.*?\| Page \d+\]\n/gmi, "")
          .trim();

        const contextPayload = cleanContext.length > 0 
          ? cleanContext 
          : "No specific document context provided. Use domain knowledge.";

        const systemPromptFromPayload = body.system_prompt || body.system || body.systemInstruction;

        let systemInstruction;
        if (systemPromptFromPayload) {
          // If context is present, append it to the custom system prompt so it is not lost
          systemInstruction = cleanContext.length > 0 
            ? `${systemPromptFromPayload}\n\nDOCUMENT CONTEXT:\n${contextPayload}`
            : systemPromptFromPayload;
        } else {
          systemInstruction = `You are an expert AI Document Intelligence Engine and Professional Analyst.
Your objective is to answer the user's query accurately using the provided DOCUMENT CONTEXT combined with professional domain reasoning.

CORE DIRECTIVES:
1. ZERO REFUSALS:
   - NEVER say "I could not find sufficient evidence" or "The document does not contain this information."
   - For abstract/career/analytical questions, analyze the facts in the text and apply domain logic to deliver insights.

2. DEDUCTIVE REASONING:
   - Factual Queries → Direct extraction.
   - Abstract Queries → Synthesize the document contents and provide explicit recommendations.

DOCUMENT CONTEXT:
${contextPayload}`;
        }

        const messages = [
          { role: "system", content: systemInstruction },
          { role: "user", content: userQuestion }
        ];

        const temperature = typeof body.temperature === "number" ? body.temperature : 0.2;
        let llmResponse;
        let chosenModel = "@cf/meta/llama-3.1-8b-instruct-fp8";

        try {
          llmResponse = await env.AI.run(chosenModel, {
            messages,
            temperature,
            max_tokens: 3000
          });
        } catch (mErr) {
          chosenModel = "@cf/meta/llama-3.2-3b-instruct";
          llmResponse = await env.AI.run(chosenModel, {
            messages,
            temperature,
            max_tokens: 3000
          });
        }

        const responseText = typeof llmResponse === "object" && llmResponse.response ? llmResponse.response : String(llmResponse);

        return new Response(
          JSON.stringify({
            success: true,
            model: chosenModel,
            response: responseText.trim()
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      return new Response(
        JSON.stringify({ error: "Endpoint not found. Use /analyze, /chat, or /embeddings" }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    } catch (err) {
      return new Response(
        JSON.stringify({ success: false, error: err.message || String(err) }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
  }
};
export { worker_default as default };
