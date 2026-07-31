/**
 * Cloudflare Worker for Master AI Document Analysis & BGE Large Embeddings
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3.1-8b-instruct
 * Formatted for Fluid, ChatGPT-Style Paragraph Responses
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

      // 2. CHATGPT-STYLE FLUID PARAGRAPH ANALYSIS ENDPOINT (@cf/meta/llama-3.1-8b-instruct)
      if (url.pathname === "/analyze" || url.pathname === "/chat" || url.pathname === "/") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const { text = "", title = "", query = "", system_prompt = "", prompt = "" } = body;

        const chatGptSystemPrompt = system_prompt || `You are an expert AI Document Assistant.
Your goal is to provide comprehensive, detailed, and highly accurate explanations formatted in complete, well-written paragraphs like ChatGPT.
Explain concepts thoroughly, synthesize facts into fluent narrative prose, and cite page numbers naturally within the text.
Deliver rich, informative responses that answer the user's question completely based on the provided document excerpts.`;

        const userQuestion = query || prompt || text;
        const contextContent = text || "";

        const messages = [
          {
            role: "system",
            content: `${chatGptSystemPrompt}\n\nDOCUMENT CONTEXT:\n${contextContent}`,
          },
          {
            role: "user",
            content: `Based on the provided document context, write a thorough, accurate, and detailed multi-paragraph response explaining:\n\n"${userQuestion}"`,
          },
        ];

        let llmResponse;
        try {
          llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
            messages: messages,
            temperature: 0.2,
            max_tokens: 2500,
          });
        } catch (mErr) {
          llmResponse = await env.AI.run("@cf/meta/llama-3-8b-instruct", {
            messages: messages,
            temperature: 0.2,
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
