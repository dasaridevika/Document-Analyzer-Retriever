/**
 * Production-Grade Cloudflare Worker AI Endpoint
 * Powered by @cf/zai-org/glm-4.7-flash & @cf/baai/bge-m3
 * Intent-Driven Document Search, Reranking & Response Synthesis
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
      // 0. QUERY UNDERSTANDING / REWRITE ENDPOINT
      // =========================================================================
      if (url.pathname === "/understand" || url.pathname === "/rewrite") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const query = body.query || body.prompt || "";
        const chatHistory = body.chat_history || [];

        const prompt = `You are a query routing and rewrite engine.
Analyze the user's query and the conversation history to classify the query intent, resolve any conversational pronouns, and rewrite the query to be a self-contained search query.

Query: "${query}"
Chat History: ${JSON.stringify(chatHistory)}

Respond in strict JSON format:
{
  "intent": "document_qa",
  "rewritten_query": "self-contained search query",
  "clarification_needed": false,
  "clarification_question": ""
}

Rules:
1. "intent" must be exactly one of: document_qa, summary, definition, comparison, extractive, follow_up, general, or ambiguous.
2. If the query is ambiguous, vague, or too short (e.g. a single verb like "list" or "compare" without context), set "clarification_needed" to true and provide a short clarifying question in "clarification_question". Otherwise, set "clarification_needed" to false.
3. "rewritten_query" should be a clear, standalone search query containing all necessary keywords from the query and history.`;

        let result;
        let chosenModel = "@cf/zai-org/glm-4.7-flash";
        try {
          result = await env.AI.run("@cf/zai-org/glm-4.7-flash", {
            messages: [
              { role: "system", content: "You respond ONLY with raw JSON." },
              { role: "user", content: prompt }
            ]
          });
        } catch (err) {
          chosenModel = "@cf/meta/llama-3.1-8b-instruct-fp8";
          try {
            result = await env.AI.run("@cf/meta/llama-3.1-8b-instruct-fp8", {
              messages: [
                { role: "system", content: "You respond ONLY with raw JSON." },
                { role: "user", content: prompt }
              ]
            });
          } catch (err2) {
            chosenModel = "@cf/meta/llama-3.2-3b-instruct";
            result = await env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
              messages: [
                { role: "system", content: "You respond ONLY with raw JSON." },
                { role: "user", content: prompt }
              ]
            });
          }
        }

        const responseText = (typeof result === 'object' && result.response) ? result.response : String(result);
        return new Response(
          JSON.stringify({
            success: true,
            model: chosenModel,
            response: responseText.trim(),
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // =========================================================================
      // 1. EMBEDDINGS ENDPOINT (@cf/baai/bge-m3)
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

        let embeddings;
        let chosenModel = "@cf/baai/bge-m3";
        try {
          embeddings = await env.AI.run("@cf/baai/bge-m3", {
            text: textList,
          });
        } catch (embErr) {
          chosenModel = "@cf/qwen/qwen3-embedding-0.6b";
          embeddings = await env.AI.run("@cf/qwen/qwen3-embedding-0.6b", {
            text: textList,
          });
        }

        return new Response(
          JSON.stringify({
            success: true,
            model: chosenModel,
            data: embeddings.data || embeddings,
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // =========================================================================
      // 2. RERANK ENDPOINT (@cf/baai/bge-reranker-base)
      // =========================================================================
      if (url.pathname === "/rerank") {
        if (request.method !== "POST") {
          return new Response(JSON.stringify({ error: "Method not allowed" }), {
            status: 405,
            headers: { ...corsHeaders, "Content-Type": "application/json" },
          });
        }

        const body = await request.json();
        const query = body.query;
        const documents = body.documents;

        if (!query || !documents) {
          return new Response(
            JSON.stringify({ error: "Missing 'query' or 'documents' parameter" }),
            { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
          );
        }

        const rerankResult = await env.AI.run("@cf/baai/bge-reranker-base", {
          query: query,
          documents: documents,
        });

        return new Response(
          JSON.stringify({
            success: true,
            model: "@cf/baai/bge-reranker-base",
            data: rerankResult.data || rerankResult,
          }),
          { headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }

      // =========================================================================
      // 3. INTENT CLASSIFICATION & ANSWER GENERATION ENDPOINT
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
              service: "DocAnalyzer Workers AI Pipeline",
              router_model: "@cf/zai-org/glm-4.7-flash",
              generation_model: "@cf/zai-org/glm-4.7-flash",
              embedding_model: "@cf/baai/bge-m3",
              reranker_model: "@cf/baai/bge-reranker-base",
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
        const userQuestion = String(body.query || body.prompt || body.question || body.message || (typeof body.text === 'string' ? body.text : "") || "Summarize").trim();
        const chatHistory = body.chat_history || [];
        const rawContext = body.context || body.text || body.document || body.contents || "";
        
        let contextString = "";
        if (Array.isArray(rawContext)) {
          contextString = rawContext.map(c => typeof c === 'object' ? (c.text || JSON.stringify(c)) : String(c)).join("\n\n---\n\n");
        } else {
          contextString = String(rawContext);
        }

        // 1. Query Routing & Intent Classification
        let intent = "document_qa";
        let rewrittenQuery = userQuestion;

        try {
          const routerPrompt = `You are a query router and intent classifier.
Analyze the user's query and the chat history to classify the query intent and rewrite it if necessary.

Query: "${userQuestion}"
Chat History: ${JSON.stringify(chatHistory)}

Classify the query into exactly one of these categories:
- document_qa: factual question about the document.
- summary: request for a summary of the document.
- definition: request for a definition of a term.
- comparison: request to compare two or more concepts.
- extractive: request for exact names, dates, or numbers.
- follow_up: query referencing previous conversation turns.
- general: query not specific to the document.
- ambiguous: query is vague, short, or unclear.

If the query is vague, short, or ambiguous, infer the intent using the chat history or general document context.
Rewrite the query to be a self-contained search query.

Respond in strict JSON format:
{
  "intent": "document_qa",
  "rewritten_query": "self-contained search query"
}`;

          const routerResponse = await env.AI.run("@cf/zai-org/glm-4.7-flash", {
            messages: [
              { role: "system", content: "You respond ONLY with raw JSON." },
              { role: "user", content: routerPrompt }
            ]
          });
          const routerText = (routerResponse.response || String(routerResponse)).trim();
          const match = routerText.match(/\{.*\}/s);
          if (match) {
            const parsed = JSON.parse(match[0]);
            if (parsed.intent) intent = parsed.intent;
            if (parsed.rewritten_query) rewrittenQuery = parsed.rewritten_query;
          }
        } catch (rErr) {
          console.error("Query routing failed, using defaults:", rErr);
        }

        // 2. Rerank top retrieved chunks with @cf/baai/bge-reranker-base
        let finalContext = contextString;
        const rawChunks = contextString.split(/\n+---\n+/).map(c => c.trim()).filter(Boolean);
        if (rawChunks.length > 1) {
          try {
            const rerankResult = await env.AI.run("@cf/baai/bge-reranker-base", {
              query: rewrittenQuery,
              documents: rawChunks,
            });
            const scoredChunks = rerankResult.data || rerankResult;
            if (Array.isArray(scoredChunks)) {
              const sortedIndices = scoredChunks
                .map((item, idx) => ({ idx, score: item.score }))
                .sort((a, b) => b.score - a.score);
              const topChunks = sortedIndices.slice(0, 6).map(item => rawChunks[item.idx]);
              finalContext = topChunks.join("\n\n---\n\n");
            }
          } catch (rerankErr) {
            console.error("Worker reranking failed:", rerankErr);
          }
        }

        // 3. Final Answer Generation (glm-4.7-flash, fallback to glm-5.2)
        const systemPrompt = `You are a document question-answering assistant.
Your job is to answer the user's query using the uploaded document as the primary source.

Rules:
1. Understand the user's intent.
2. Find the most relevant content in the document and answer directly.
3. Prefer exact sentences from the document for definitions or factual answers.
4. For broader queries, synthesize a concise answer from multiple relevant parts.
5. If no strong context exists or the answer is not present in the document, say "The answer is not clearly present in the document." instead of hallucinating.
6. Always cite page numbers or chunk IDs.
7. Start your response directly with the answer.

DOCUMENT CONTEXT:
${finalContext}`;

        const finalMessages = [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Question: ${rewrittenQuery}` }
        ];

        const temperature = typeof body.temperature === 'number' ? body.temperature : 0.1;

        let llmResponse;
        let chosenModel = "@cf/zai-org/glm-4.7-flash";
        try {
          llmResponse = await env.AI.run("@cf/zai-org/glm-4.7-flash", {
            messages: finalMessages,
            temperature: temperature,
            max_tokens: 3000,
          });
        } catch (mErr) {
          chosenModel = "@cf/zai-org/glm-5.2";
          try {
            llmResponse = await env.AI.run("@cf/zai-org/glm-5.2", {
              messages: finalMessages,
              temperature: temperature,
              max_tokens: 3000,
            });
          } catch (mErr2) {
            chosenModel = "@cf/meta/llama-3.1-8b-instruct-fp8";
            llmResponse = await env.AI.run("@cf/meta/llama-3.1-8b-instruct-fp8", {
              messages: finalMessages,
              temperature: temperature,
              max_tokens: 3000,
            });
          }
        }

        const responseText = (typeof llmResponse === 'object' && llmResponse.response) 
          ? llmResponse.response 
          : String(llmResponse);

        return new Response(
          JSON.stringify({
            success: true,
            model: chosenModel,
            intent: intent,
            rewritten_query: rewrittenQuery,
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
