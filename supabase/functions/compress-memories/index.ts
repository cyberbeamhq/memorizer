// Supabase Edge Function: Compress Memories
// Runs background compression jobs on old memories

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface CompressionJob {
  id: string
  memory_id: string
  user_id: string
  policy_name: string
  retry_count: number
  max_retries: number
}

interface Memory {
  id: string
  content: string
  tier: string
  metadata: any
}

serve(async (req) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Create Supabase client with service role (bypasses RLS)
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '',
      {
        auth: {
          autoRefreshToken: false,
          persistSession: false
        }
      }
    )

    // Get pending compression jobs
    const { data: jobs, error: jobsError } = await supabaseClient
      .from('compression_jobs')
      .select('*')
      .eq('status', 'pending')
      .lte('next_retry_at', new Date().toISOString())
      .lt('retry_count', 3)
      .limit(10)

    if (jobsError) {
      throw jobsError
    }

    if (!jobs || jobs.length === 0) {
      return new Response(
        JSON.stringify({ message: 'No pending jobs', processed: 0 }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const results = []

    // Process each job
    for (const job of jobs as CompressionJob[]) {
      try {
        // Mark as processing
        await supabaseClient
          .from('compression_jobs')
          .update({ status: 'processing' })
          .eq('id', job.id)

        // Get the memory
        const { data: memory, error: memError } = await supabaseClient
          .from('memories')
          .select('*')
          .eq('id', job.memory_id)
          .single()

        if (memError || !memory) {
          throw new Error(`Memory not found: ${job.memory_id}`)
        }

        // Perform compression based on policy
        const compressed = await compressMemory(memory as Memory, job.policy_name)

        // Update memory with compressed version
        const { error: updateError } = await supabaseClient
          .from('memories')
          .update({
            compressed_content: compressed.content,
            compression_ratio: compressed.ratio,
            compression_algorithm: compressed.algorithm,
            tier: compressed.tier || memory.tier
          })
          .eq('id', job.memory_id)

        if (updateError) {
          throw updateError
        }

        // Mark job as completed
        await supabaseClient
          .from('compression_jobs')
          .update({
            status: 'completed',
            completed_at: new Date().toISOString(),
            original_size: memory.content.length,
            compressed_size: compressed.content.length,
            compression_ratio: compressed.ratio
          })
          .eq('id', job.id)

        results.push({ job_id: job.id, success: true })

      } catch (error) {
        // Handle job failure
        const newRetryCount = job.retry_count + 1
        const nextRetryAt = new Date(Date.now() + Math.pow(2, newRetryCount) * 60000) // Exponential backoff

        if (newRetryCount >= job.max_retries) {
          // Max retries reached, mark as failed
          await supabaseClient
            .from('compression_jobs')
            .update({
              status: 'failed',
              error_message: error.message
            })
            .eq('id', job.id)
        } else {
          // Retry later
          await supabaseClient
            .from('compression_jobs')
            .update({
              status: 'pending',
              retry_count: newRetryCount,
              next_retry_at: nextRetryAt.toISOString(),
              error_message: error.message
            })
            .eq('id', job.id)
        }

        results.push({ job_id: job.id, success: false, error: error.message })
      }
    }

    return new Response(
      JSON.stringify({
        message: 'Compression jobs processed',
        processed: results.length,
        results
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
    )
  }
})

// Compression logic
async function compressMemory(
  memory: Memory,
  policyName: string
): Promise<{ content: string; ratio: number; algorithm: string; tier?: string }> {
  const content = memory.content
  const originalLength = content.length

  // Simple text compression for demo
  // In production, you'd call an LLM API or use gzip
  let compressed: string
  let algorithm: string
  let newTier: string | undefined

  switch (policyName) {
    case 'age_based_gzip':
      // Simulate gzip compression
      compressed = await gzipCompress(content)
      algorithm = 'gzip'
      newTier = 'mid_term'
      break

    case 'tier_based_adaptive':
      // LLM-based semantic compression
      compressed = await llmCompress(content, 0.7) // 70% of original
      algorithm = 'llm_semantic'
      newTier = 'long_term'
      break

    case 'size_based_zlib':
      // Simulate zlib compression
      compressed = await zlibCompress(content)
      algorithm = 'zlib'
      break

    default:
      // Simple summarization
      compressed = content.substring(0, Math.floor(content.length * 0.8))
      algorithm = 'simple'
  }

  const compressedLength = compressed.length
  const ratio = compressedLength / originalLength

  return {
    content: compressed,
    ratio,
    algorithm,
    tier: newTier
  }
}

// Compression helpers
async function gzipCompress(text: string): Promise<string> {
  // In real implementation, use actual gzip
  // For now, return a truncated version as demo
  return text.substring(0, Math.floor(text.length * 0.6))
}

async function zlibCompress(text: string): Promise<string> {
  // Similar to gzip
  return text.substring(0, Math.floor(text.length * 0.65))
}

async function llmCompress(text: string, targetRatio: number): Promise<string> {
  // Call OpenAI or other LLM for semantic compression
  // For demo, return truncated text
  const targetLength = Math.floor(text.length * targetRatio)

  // In production, you would do:
  // const response = await fetch('https://api.openai.com/v1/chat/completions', {
  //   method: 'POST',
  //   headers: {
  //     'Authorization': `Bearer ${Deno.env.get('OPENAI_API_KEY')}`,
  //     'Content-Type': 'application/json',
  //   },
  //   body: JSON.stringify({
  //     model: 'gpt-3.5-turbo',
  //     messages: [{
  //       role: 'user',
  //       content: `Summarize to ${Math.floor(targetRatio * 100)}% length: ${text}`
  //     }]
  //   })
  // })
  // const data = await response.json()
  // return data.choices[0].message.content

  return text.substring(0, targetLength)
}

/* To deploy this function:

1. Install Supabase CLI:
   npm install -g supabase

2. Login to Supabase:
   supabase login

3. Link to your project:
   supabase link --project-ref your-project-ref

4. Deploy the function:
   supabase functions deploy compress-memories

5. Set environment variables:
   supabase secrets set OPENAI_API_KEY=your-key

6. Schedule with pg_cron (in Supabase SQL editor):
   SELECT cron.schedule(
     'compress-memories-hourly',
     '0 * * * *',
     $$
     SELECT net.http_post(
       url := 'https://your-project.supabase.co/functions/v1/compress-memories',
       headers := '{"Authorization": "Bearer YOUR_ANON_KEY"}'::jsonb
     );
     $$
   );
*/
