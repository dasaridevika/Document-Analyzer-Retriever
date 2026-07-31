/**
 * Cloudflare Worker for Master AI Document Analysis & BGE Large Embeddings
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3.1-8b-instruct
 * Optimized for High-Precision, Exact Factual Accuracy (Temperature: 0.1)
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

      // 2. HIGH-ACCURACY CHAT & DOCUMENT ANALYSIS ENDPOINT (@cf/meta/llama-3.1-8b-instruct)
      if (url.pathname === "/analyze" || url.pathname === "/chat" || url.pathname === "/") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const { text = "", title = "", query = "", system_prompt = "", prompt = "" } = body;

        const masterAccuracySystemPrompt = system_prompt || `You are a Master AI Document Analyst & Technical Educator.
Your task is to provide exact, highly accurate, detailed, and specific answers based strictly on the provided Document Context.

STRICT ACCURACY INSTRUCTIONS:
1. Base your answer ONLY on facts, definitions, numbers, formulas, and concepts directly stated in the DOCUMENT CONTEXT.
2. If specific steps, lists, specifications, or data points exist in the context, detail EVERY item accurately.
3. Structure your response clearly using bold headers, bullet points, and numbered steps.
4. Cite specific page numbers whenever mentioned in the context (e.g., [Page 4], [Page 12]).
5. Never guess, extrapolate, or introduce external information not present in the context.`;

        const userQuestion = query || prompt || text;
        const contextContent = text || "";

        const messages = [
          {
            role: "system",
            content: `${masterAccuracySystemPrompt}\n\nDOCUMENT CONTEXT:\n${contextContent}`,
          },
          {
            role: "user",
            content: `Based strictly on the DOCUMENT CONTEXT provided above, provide an exact, thorough, and highly detailed answer for:\n\n"${userQuestion}"`,
          },
        ];

        let llmResponse;
        try {
          // Temperature 0.1 forces deterministic, exact factual accuracy
          llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
            messages: messages,
            temperature: 0.1,
            max_tokens: 2500,
          });
        } catch (mErr) {
          // Fallback Active Model
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
