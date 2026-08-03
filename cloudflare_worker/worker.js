/**
 * Cloudflare Worker for Production-Grade Master AI Document Analysis
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3.1-8b-instruct
 * Intelligent Intent Classification & Executive Summary Formatting
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

      // 2. PRODUCTION LLM ENDPOINT (@cf/meta/llama-3.1-8b-instruct)
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
        const { text = "", title = "", query = "", system_prompt = "", prompt = "", is_broad = false, temperature = 0.1 } = body;
        const userQuestion = query || prompt || text;
        const contextContent = text || "";

        const lowerQ = userQuestion.toLowerCase().trim();
        const isBroadQuery = is_broad || [
          "explain contents", "explain the contents", "explain the pdf", "contents", "summarize", "summary",
          "overview", "what is in", "tell me about", "full document", "complete details", "describe"
        ].some(k => lowerQ.includes(k));

        let querySpecificPrompt;
        if (isBroadQuery) {
          querySpecificPrompt = `You are a Senior Technical Document Lead.
The user requested a comprehensive explanation of the document contents: "${userQuestion}".

PRODUCE A PRODUCTION-GRADE EXECUTIVE SUMMARY STRUCTURED WITH THE FOLLOWING SECTIONS:
1. **Executive Overview & Objective**: Explain the main purpose and core subject of the document.
2. **Key Sections & Topics Covered**: Provide a structured breakdown of major modules, rules, or chapters spanning the pages.
3. **Core Technical Details & Specifications**: Highlight important rules, formulas, components, or requirements found in the context.
4. **Conclusion & Key Takeaways**: Summarize the final conclusions.

RULES:
- Cite page numbers naturally in brackets like [Page 1], [Page 7].
- Do NOT output raw chunk headers (do NOT print "Section (Page 1):"). Write fluent, professional prose and bullet points.`;
        } else {
          querySpecificPrompt = `You are a Master AI Document Assistant.
CRITICAL MANDATE:
Answer ONLY the specific topic asked in the user query: "${userQuestion}".

STRICT QUERY ISOLATION RULES:
1. Explain ONLY what is explicitly asked in "${userQuestion}".
2. Do NOT mention, summarize, or list unrelated topics present in the document context.
3. Provide exact definitions, syntax, code examples, rules, methods, and page citations matching "${userQuestion}".
4. Write in clear, professional paragraphs and structured bullet points. Cite page numbers like [Page 8].
5. Base your response strictly on the provided DOCUMENT CONTEXT.`;
        }

        const messages = [
          {
            role: "system",
            content: `${querySpecificPrompt}\n\nDOCUMENT CONTEXT:\n${contextContent}`,
          },
          {
            role: "user",
            content: `Based strictly on the DOCUMENT CONTEXT provided above, write a production-grade, highly accurate response answering:\n\n"${userQuestion}"`,
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
