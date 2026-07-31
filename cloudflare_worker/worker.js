/**
 * Cloudflare Worker for Document Analysis & BGE Large Embeddings
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3-8b-instruct
 */

export default {
  async fetch(request, env, ctx) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    const url = new URL(request.url);

    try {
      // 1. EMBEDDINGS ENDPOINT (@cf/baai/bge-large-en-v1.5)
      if (url.pathname === "/embeddings" || url.pathname === "/embed") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const textInput = body.text || body.input;

        if (!textInput) {
          return new Response(
            JSON.stringify({ error: "Missing 'text' parameter" }),
            { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
          );
        }

        const textList = Array.isArray(textInput) ? textInput : [textInput];

        const embeddings = await env.AI.run("@cf/baai/bge-large-en-v1.5", {
          text: textList,
        });

        return new Response(
          JSON.stringify({
            success: true,
            model: "@cf/baai/bge-large-en-v1.5",
            data: embeddings.data || embeddings,
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // 2. CHAT & DOCUMENT ANALYSIS ENDPOINT (@cf/meta/llama-3-8b-instruct - ChatGPT Detailed Style)
      if (url.pathname === "/analyze" || url.pathname === "/chat" || url.pathname === "/") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const { text = "", title = "", query = "", system_prompt = "", prompt = "" } = body;

        const effectiveSystemPrompt = system_prompt || `You are an expert AI Document Assistant.
Your goal is to provide comprehensive, detailed, and thoroughly structured explanations like ChatGPT.
Break down complex topics into clear sections, use bullet points, bold key concepts, and cite page numbers whenever mentioned in the context.
Deliver rich, informative responses that answer the user's question completely based on the provided document excerpts.`;

        const userMessage = query || prompt || text;
        const contextText = text || "";

        const messages = [
          { role: "system", content: `${effectiveSystemPrompt}\n\nDOCUMENT CONTEXT:\n${contextText}` },
          { role: "user", content: userMessage },
        ];

        // Run Llama-3 8B Instruct model with high token limit for detailed responses
        const llmResponse = await env.AI.run("@cf/meta/llama-3-8b-instruct", {
          messages: messages,
          temperature: 0.3,
          max_tokens: 2500,
        });

        return new Response(
          JSON.stringify({
            success: true,
            model: "@cf/meta/llama-3-8b-instruct",
            result: llmResponse,
            response: llmResponse.response || llmResponse,
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      return new Response(
        JSON.stringify({ error: "Endpoint not found. Use /analyze or /embeddings" }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    } catch (err) {
      return new Response(
        JSON.stringify({ success: false, error: err.message || String(err) }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
  },
};
