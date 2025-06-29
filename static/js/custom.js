document.addEventListener('DOMContentLoaded', function() {
  var analyzeForm = document.querySelector('form[action="/predict"]');
  if (analyzeForm) {
    analyzeForm.addEventListener('submit', function() {
      document.getElementById('loading-overlay').style.display = 'flex';
    });
  }
});

document.addEventListener('DOMContentLoaded', function() {
  var alertEl = document.querySelector('.custom-alert');
    if (alertEl) {
      setTimeout(function() {
        alertEl.style.display = 'none';
      }, 4000); // 4 seconds
    }
});

document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.remove-result-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      // Remove the closest .result-card
      btn.closest('.result-card').remove();
    });
  });
});