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

        const systemPromptFromPayload = body.system_prompt || body.system || body.systemInstruction;

        let systemInstruction;
        if (systemPromptFromPayload) {
          systemInstruction = systemPromptFromPayload;
        } else {
          if (isBroadQuery) {
            systemInstruction = `You are an expert AI Document Assistant and Career/Domain Analyst.
The user requested a clear summary and natural explanation of the document: "${userQuestion}".

PRODUCE A WELL-WRITTEN EXECUTIVE OVERVIEW WITH THESE SECTIONS:
1. **Document Purpose & Overview**: Explain the subject of the document clearly in natural paragraphs.
2. **Key Positions, Figures & Details**: Highlight specific roles, dates, numbers, requirements, or locations mentioned.
3. **Summary & Key Takeaways**: Provide a concise conclusion.

CRITICAL DIRECTIVES:
1. NEVER output phrases like "I could not find sufficient evidence to answer this question in the uploaded document."
2. Even if a question requires deduction (e.g., inferring location from phone codes/cities, inferring job role from technical skills, evaluating a PRD), use domain logic to provide a thoughtful, helpful answer.

DOCUMENT CONTEXT:
${cleanContext}`;
          } else {
            systemInstruction = `You are an expert AI Document Assistant and Career/Domain Analyst.
Your primary objective is to address the user's true intent accurately and adaptively using the provided DOCUMENT CONTEXT.

CRITICAL DIRECTIVES:
1. NEVER output phrases like "I could not find sufficient evidence to answer this question in the uploaded document."
2. Even if a question requires deduction (e.g., inferring location from phone codes/cities, inferring job role from technical skills, evaluating a PRD), synthesize the text and use domain logic to provide a thoughtful, helpful answer.
3. For direct questions, state the fact directly from the context.

DOCUMENT CONTEXT:
${cleanContext}`;
          }
        }

        const userContent = systemPromptFromPayload
          ? userQuestion
          : `You must answer the user question using ONLY the provided verified document context.\n\n<DOCUMENT_CONTEXT>\n${cleanContext}\n</DOCUMENT_CONTEXT>\n\nQuestion: "${userQuestion}"`;

        const messages = [
          {
            role: "system",
            content: systemInstruction,
          },
          {
            role: "user",
            content: userContent,
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
