/**
 * Production-Grade Cloudflare Worker AI Endpoint
 * Powered by @cf/baai/bge-large-en-v1.5 & @cf/meta/llama-3.1-8b-instruct
 * Intent-Driven Document Search & Detailed Response Synthesis
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
      // 2. QUERY-SPECIFIC INTENT LLM ENDPOINT (@cf/meta/llama-3.1-8b-instruct)
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
        const userQuestion = String(body.query || body.prompt || body.question || body.message || (typeof body.text === 'string' ? body.text : "") || "Summarize document").trim();
        
        let rawContext = body.context || body.text || body.document || body.contents || "";
        if (Array.isArray(rawContext)) {
          rawContext = rawContext.map(c => typeof c === 'object' ? (c.text || JSON.stringify(c)) : String(c)).join("\n\n");
        }

        const cleanContext = String(rawContext)
          .replace(/Visual\s*\[Page\s*\d+\]\s*Visual/gi, "")
          .replace(/^\s*Visual\s*$/gmi, "")
          .replace(/^\s*Page\s*\d+\s*\[Page\s*\d+\]\s*/gmi, "")
          .replace(/^\[Document:.*?\| Page \d+\]\n/gmi, "")
          .trim();

        const lowerQ = userQuestion.toLowerCase().trim();
        const isBroadQuery = body.is_broad || [
          "explain contents", "explain the contents", "explain the pdf", "contents",
          "summarize", "summary", "overview", "what is in", "tell me about",
          "full document", "complete details", "describe", "table of contents",
          "what is it all about", "what is it about", "what is this about", "summarize it"
        ].some(k => lowerQ.includes(k));

        let systemInstruction;
        if (isBroadQuery) {
          systemInstruction = `You are a Lead AI Document Analyst.
The user requested a clear summary and natural explanation of the document: "${userQuestion}".

PRODUCE A WELL-WRITTEN EXECUTIVE OVERVIEW WITH THESE SECTIONS:
1. **Document Purpose & Overview**: Explain the subject of the document clearly in natural paragraphs.
2. **Key Positions, Figures & Details**: Highlight specific roles, dates, numbers, requirements, or locations mentioned.
3. **Summary & Key Takeaways**: Provide a concise conclusion.

RULES:
- Base your response strictly on the DOCUMENT CONTEXT.
- Write fluent, natural English prose and bullet points. Do NOT print raw chunk tags.
- Cite page numbers naturally like [Page 1], [Page 2].`;
        } else {
          systemInstruction = `You are a Master AI Document Assistant.
CRITICAL INSTRUCTION:
Provide a comprehensive, highly accurate, and detail-specific answer that directly fulfills the user's intent: "${userQuestion}".

INTENT MATCHING RULES:
1. Understand the core question and intent behind "${userQuestion}".
2. Extract ALL exact figures, numbers, definitions, rules, conditions, formulas, and steps provided in the DOCUMENT CONTEXT that answer this intent.
3. Structure your response cleanly using headers, bold key terms, and bullet points.
4. Cite page numbers naturally like [Page X] for every fact stated.
5. Base your answer strictly on the DOCUMENT CONTEXT provided below.`;
        }

        const messages = [
          {
            role: "system",
            content: systemInstruction,
          },
          {
            role: "user",
            content: `You must answer the user question using ONLY the provided verified document context.\n\n<DOCUMENT_CONTEXT>\n${cleanContext}\n</DOCUMENT_CONTEXT>\n\nQuestion: "${userQuestion}"`,
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
          llmResponse = await env.AI.run("@cf/meta/llama-8b-instruct", {
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
