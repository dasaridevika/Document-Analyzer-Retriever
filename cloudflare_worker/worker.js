/**
 * Production-Grade Cloudflare Worker AI Endpoint
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3.1-8b-instruct
 * Optimized for Any Uploaded PDF & Exact Query Requirements
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
      // =========================================================================
      // 1. EMBEDDINGS ENDPOINT (@cf/baai/bge-large-en-v1.5)
      // =========================================================================
      if (url.pathname === "/embeddings" || url.pathname === "/embed") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
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

      // =========================================================================
      // 2. QUERY-SPECIFIC LLM ENDPOINT (@cf/meta/llama-3.1-8b-instruct)
      // =========================================================================
      if (
        url.pathname === "/analyze" ||
        url.pathname === "/chat" ||
        url.pathname === "/" ||
        url.pathname === ""
      ) {
        if (request.method === "GET") {
          return new Response(
            JSON.stringify({
              status: "online",
              service: "DocAnalyzer Cloudflare Workers AI Endpoint",
              llm_model: "@cf/meta/llama-3.1-8b-instruct",
              embedding_model: "@cf/baai/bge-large-en-v1.5",
            }),
            {
              status: 200,
              headers: { ...corsHeaders, "Content-Type": "application/json" },
            }
          );
        }

        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();

        // Robust Field Resolution
        const userQuestion = (body.query || body.prompt || body.question || body.message || body.text || "Summarize the document").trim();
        
        let rawContext = body.context || body.text || body.document || body.contents || "";
        if (Array.isArray(rawContext)) {
          rawContext = rawContext.map(c => typeof c === 'object' ? (c.text || JSON.stringify(c)) : String(c)).join("\n\n");
        }

        // Clean Context from OCR noise artifacts
        const cleanContext = String(rawContext)
          .replace(/Visual\s*\[Page\s*\d+\]\s*Visual/gi, "")
          .replace(/^\s*Visual\s*$/gmi, "")
          .replace(/^\s*Page\s*\d+\s*\[Page\s*\d+\]\s*/gmi, "")
          .trim();

        const lowerQ = userQuestion.toLowerCase().trim();
        const isBroadQuery = body.is_broad || [
          "explain contents", "explain the contents", "explain the pdf", "contents",
          "summarize", "summary", "overview", "what is in", "tell me about",
          "full document", "complete details", "describe", "table of contents"
        ].some(k => lowerQ.includes(k));

        let systemInstruction;
        if (isBroadQuery) {
          systemInstruction = `You are an Expert Lead Document Analyst.
The user requested a complete explanation of the document contents: "${userQuestion}".

PRODUCE A PRODUCTION-GRADE EXECUTIVE SUMMARY WITH THESE EXACT SECTIONS:
1. **Executive Overview & Primary Objective**: Summarize the core purpose and subject of the document.
2. **Key Sections & Topics Covered**: Provide a detailed, organized breakdown of major modules, chapters, rules, or topics.
3. **Core Technical Details & Specifications**: Detail exact figures, numbers, formulas, rules, requirements, or data.
4. **Conclusion & Key Takeaways**: Summarize the primary conclusions.

RULES:
- Base your response strictly on the provided DOCUMENT CONTEXT.
- Cite page numbers naturally like [Page 1], [Page 4].
- Do NOT output raw chunk headers (never print "Section (Page 1):"). Write fluent, professional markdown prose and bullet points.`;
        } else {
          systemInstruction = `You are a Master AI Document Assistant.
CRITICAL MANDATE:
Answer ONLY the specific query asked by the user: "${userQuestion}".

STRICT QUERY ISOLATION RULES:
1. Focus SPECIFICALLY and ONLY on answering "${userQuestion}".
2. Explain exact definitions, syntax, rules, code examples, steps, formulas, and methods matching "${userQuestion}".
3. Do NOT mention, summarize, or list unrelated sections present in the document context.
4. Cite page numbers naturally like [Page 4], [Page 12].
5. Base your response strictly on the provided DOCUMENT CONTEXT.`;
        }

        const messages = [
          {
            role: "system",
            content: `${systemInstruction}\n\nDOCUMENT CONTEXT:\n${cleanContext}`,
          },
          {
            role: "user",
            content: `Based strictly on the DOCUMENT CONTEXT provided above, write a direct, highly accurate response answering:\n\n"${userQuestion}"`,
          },
        ];

        const temperature = typeof body.temperature === 'number' ? body.temperature : 0.1;

        let llmResponse;
        try {
          llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
            messages: messages,
            temperature: temperature,
            max_tokens: 3000,
          });
        } catch (mErr) {
          llmResponse = await env.AI.run("@cf/meta/llama-3-8b-instruct", {
            messages: messages,
            temperature: temperature,
            max_tokens: 3000,
          });
        }

        const responseText = (typeof llmResponse === 'object' && llmResponse.response) 
          ? llmResponse.response 
          : String(llmResponse);

        return new Response(
          JSON.stringify({
            success: true,
            model: "@cf/meta/llama-3.1-8b-instruct",
            result: llmResponse,
            response: responseText.trim(),
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
