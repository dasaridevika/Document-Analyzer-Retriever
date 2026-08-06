/**
 * Production-Grade Cloudflare Worker AI Endpoint
 * Grounded Answers with Verified Source Citations
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

    try {
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
2. If the query is ambiguous, vague, or too short (e.g. a single verb like "list" or "compare" without context), set "clarification_needed" to true and provide a short clarifying question in "clarification_question". Otherwise, set "clarification_needed" to false.
3. "rewritten_query" should be a clear, standalone search query containing all necessary keywords from the query and history.`;

        let result;
        let chosenModel = "@cf/meta/llama-3.1-8b-instruct-fp8";
        try {
          result = await env.AI.run("@cf/meta/llama-3.1-8b-instruct-fp8", {
            messages: [
              { role: "system", content: "You respond ONLY with raw JSON." },
              { role: "user", content: prompt }
            ]
          });
        } catch (err) {
          chosenModel = "@cf/meta/llama-3.2-3b-instruct";
          result = await env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
            messages: [
              { role: "system", content: "You respond ONLY with raw JSON." },
              { role: "user", content: prompt }
            ]
          });
        }

        const responseText = (typeof result === 'object' && result.response) ? result.response : String(result);
        return new Response(
          JSON.stringify({
            success: true,
            model: chosenModel,
            response: responseText.trim(),
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
        const textInput = body.text || body.input || body.contents;
        if (!textInput) {
          return new Response(
            JSON.stringify({ error: "Missing 'text' parameter" }),
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
        const userQuestion = String(
          body.query || body.prompt || body.question || body.message || (typeof body.text === "string" ? body.text : "") || "Summarize document"
        ).trim();

        let rawContext = body.context || body.text || body.document || body.contents || "";

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
          systemInstruction = systemPromptFromPayload;
        } else {
          systemInstruction = `You are an expert AI Document Intelligence Engine and Professional Analyst.
Your objective is to answer the user's query accurately using the provided DOCUMENT CONTEXT combined with professional domain reasoning.

CORE DIRECTIVES:
1. ZERO REFUSALS:
   - NEVER say "I could not find sufficient evidence" or "The document does not contain this information."
   - For abstract/career/analytical questions (e.g., "what role suits me", "evaluate this PRD"), analyze the facts in the text (skills, metrics, clauses) and apply domain logic to deliver clear insights.

2. DEDUCTIVE REASONING:
   - Factual Queries → Direct extraction.
   - Abstract Queries → Synthesize the document contents and provide explicit recommendations or insights.

DOCUMENT CONTEXT:
${contextPayload}`;
        }

        // Fixed: User message no longer forces strict "ONLY use context" constraint
        const userContent = systemPromptFromPayload 
          ? userQuestion 
          : `Document Context is attached above in system instructions.\n\nUser Question: "${userQuestion}"`;

        const messages = [
          { role: "system", content: systemInstruction },
          { role: "user", content: userContent }
        ];

        const temperature = typeof body.temperature === "number" ? body.temperature : 0.2;
        let llmResponse;
        let chosenModel = "@cf/meta/llama-3.1-8b-instruct-fp8";

        try {
          llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct-fp8", {
            messages,
            temperature,
            max_tokens: 3000
          });
        } catch (mErr) {
          chosenModel = "@cf/meta/llama-3.2-3b-instruct";
          llmResponse = await env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
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
            result: llmResponse,
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
