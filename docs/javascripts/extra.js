// Add a "copy to clipboard" button next to each code block
$(document).ready(function() {
    // https://clipboardjs.com/
    var selectors = document.querySelectorAll('pre code');
    var copyButton = '<div class="clipboard"><span class="btn btn-neutral btn-clipboard" title="Copy to clipboard"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-copy" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M4 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1z"/></svg></span></div>';
    Array.prototype.forEach.call(selectors, function(selector){
      selector.insertAdjacentHTML('beforebegin', copyButton);
    });
    var clipboard = new ClipboardJS('.btn-clipboard', {
      target: function (trigger) {
        return trigger.parentNode.nextElementSibling;
      }
    });
  
    clipboard.on('success', function (e) {
      e.clearSelection();
  
      // https://atomiks.github.io/tippyjs/v6/all-props/
      var tippyInstance = tippy(
        e.trigger,
        {
          content: 'Copied',
          showOnCreate: true,
          trigger: 'manual',
        },
      );
      setTimeout(function() { tippyInstance.hide(); }, 1000);
    });
  });