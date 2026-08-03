/**
 * Cloudflare Worker for Master AI Document Analysis & BGE Large Embeddings
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3.1-8b-instruct
 * Optimized for Direct, Detail-Specific & Accurate Responses (No Template Sections)
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

      // 2. DIRECT DETAIL-SPECIFIC LLM ENDPOINT (@cf/meta/llama-3.1-8b-instruct)
      if (url.pathname === "/analyze" || url.pathname === "/chat" || url.pathname === "/" || url.pathname === "") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ message: "Cloudflare Workers AI LLM API is Online", model: "@cf/meta/llama-3.1-8b-instruct" }), {
            status: 200,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const { text = "", title = "", query = "", system_prompt = "", prompt = "" } = body;

        const directDetailSystemPrompt = system_prompt || `You are a Master AI Document Analyst & Technical Educator.
Your task is to provide a direct, exact, highly detailed, and specific answer based strictly on the provided Document Context.

RULES FOR DIRECT & ACCURATE RESPONSES:
1. Answer the user's question DIRECTLY without artificial section titles or template intros (do NOT use "Executive Summary", "Detailed Breakdown", or "Key Takeaways").
2. Detail every relevant concept, step, definition, formula, or specification present in the context thoroughly.
3. Use bold text for key terms and clear bullet points/numbered lists where helpful.
4. Cite page numbers naturally in the text (e.g. [Page 4], [Page 12]).
5. Base your response strictly on the provided document context without making up unverified information.`;

        const userQuestion = query || prompt || text;
        const contextContent = text || "";

        const messages = [
          {
            role: "system",
            content: `${directDetailSystemPrompt}\n\nDOCUMENT CONTEXT:\n${contextContent}`,
          },
          {
            role: "user",
            content: `Based strictly on the DOCUMENT CONTEXT provided above, write a direct, highly accurate, and detail-specific answer for:\n\n"${userQuestion}"`,
          },
        ];

        let llmResponse;
        try {
          llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
            messages: messages,
            temperature: 0.1,
            max_tokens: 2500,
          });
        } catch (mErr) {
          llmResponse = await env.AI.run("@cf/meta/llama-3-8b-instruct", {
            messages: messages,
            temperature: 0.1,
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
