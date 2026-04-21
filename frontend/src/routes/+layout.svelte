<script>
  let { children } = $props();
</script>

<header>
  <a href="/"><strong>PII Redactor</strong></a>
</header>
<main>{@render children()}</main>

<style>
  :global(body) {
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: #0f1115;
    color: #e6e6e6;
  }
  header {
    padding: 12px 20px;
    border-bottom: 1px solid #222;
    background: #16181d;
  }
  header a {
    color: #e6e6e6;
    text-decoration: none;
  }
  main {
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
  }
</style>
