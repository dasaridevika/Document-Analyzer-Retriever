/**
 * Cloudflare Worker for Master AI Document Analysis & BGE Large Embeddings
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3.1-8b-instruct
 * Complete Document Scope Analysis & Direct High-Precision Answers
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

      // 2. COMPLETE DOCUMENT SCOPE ANALYSIS LLM ENDPOINT (@cf/meta/llama-3.1-8b-instruct)
      if (url.pathname === "/analyze" || url.pathname === "/chat" || url.pathname === "/" || url.pathname === "") {
        if (request.method === "GET") {
          return new Response(JSON.stringify({
            status: "online",
            service: "Cloudflare Workers AI Endpoint",
            llm_model: "@cf/meta/llama-3.1-8b-instruct",
            embedding_model: "@cf/baai/bge-large-en-v1.5"
          }), {
            status: 200,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const { text = "", title = "", query = "", system_prompt = "", prompt = "", temperature = 0.1 } = body;

        const completeDocPrompt = system_prompt || `You are a Master AI Document Analyst.
Your task is to analyze the COMPLETE document context provided below (covering the entire document scope from beginning to end) and provide a direct, highly accurate, detail-specific answer for the user's query.

STRICT INSTRUCTIONS:
1. Analyze the FULL document scope provided in the context below before answering.
2. Answer the user's query DIRECTLY without filler intros, conversational pleasantries, or artificial section headers (do NOT use "Executive Summary", "Detailed Breakdown", or "Key Takeaways").
3. Detail every relevant concept, step, definition, formula, number, or specification present across the entire document.
4. Filter out raw PDF OCR image labels like "Visual [Page 2] Visual".
5. Cite page numbers naturally in the text (e.g. [Page 4], [Page 12]).
6. Base your response strictly on the provided FULL DOCUMENT CONTEXT without making up unverified information.`;

        const userQuestion = query || prompt || text;
        const contextContent = text || "";

        const messages = [
          {
            role: "system",
            content: `${completeDocPrompt}\n\nFULL DOCUMENT CONTEXT:\n${contextContent}`,
          },
          {
            role: "user",
            content: `Based strictly on analyzing the FULL DOCUMENT CONTEXT provided above, provide a direct, highly accurate, detail-specific answer for:\n\n"${userQuestion}"`,
          },
        ];

        let llmResponse;
        try {
          llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
            messages: messages,
            temperature: temperature,
            max_tokens: 2500,
          });
        } catch (mErr) {
          llmResponse = await env.AI.run("@cf/meta/llama-3-8b-instruct", {
            messages: messages,
            temperature: temperature,
            max_tokens: 2500,
          });
        }

        const responseText = llmResponse.response || llmResponse;

        return new Response(
          JSON.stringify({
            success: true,
            model: "@cf/meta/llama-3.1-8b-instruct",
            result: llmResponse,
            response: responseText,
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
  },
};
