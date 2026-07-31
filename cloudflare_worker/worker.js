/**
 * Cloudflare Worker for Document Analysis & BGE Large Embeddings
 * Uses Cloudflare Workers AI bindings (@cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3-8b-instruct)
 */

export default {
  async fetch(request, env, ctx) {
    // Enable CORS headers
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

        // Support string or array of strings
        const textList = Array.isArray(textInput) ? textInput : [textInput];

        // Call Cloudflare Workers AI embedding model
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

      // 2. DOCUMENT ANALYSIS ENDPOINT (@cf/meta/llama-3-8b-instruct)
      if (url.pathname === "/analyze" || url.pathname === "/") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const { text, title = "", analysis_type = "summary", prompt = "" } = body;

        if (!text) {
          return new Response(
            JSON.stringify({ error: "Missing 'text' parameter for analysis" }),
            { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
          );
        }

        const systemPrompt = `You are an expert Document Analysis AI. Analyze the provided document text and output a JSON object containing EXACTLY these keys:
- "summary": String concise summary.
- "topics": Array of string topics.
- "keywords": Array of string key terms.
- "sentiment": String ("positive", "negative", or "neutral").
- "important_points": Array of key takeaways.
- "action_items": Array of actionable next steps.

Return ONLY a valid JSON object without markdown wrappers or code blocks.`;

        const userPrompt = `${prompt}\n\nDocument Title: ${title}\nContent:\n${text}`;

        const llmResponse = await env.AI.run("@cf/meta/llama-3-8b-instruct", {
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          temperature: 0.2,
          max_tokens: 1500,
        });

        return new Response(
          JSON.stringify({
            success: true,
            model: "@cf/meta/llama-3-8b-instruct",
            result: llmResponse,
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
