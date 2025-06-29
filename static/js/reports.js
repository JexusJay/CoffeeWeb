// Get data from global variable set in reports.html
const {
  labels,
  areaDataArr,
  perimeterDataArr,
  eccentricityArr,
  solidityArr
} = window.REPORTS_DATA;

// Chart.js datasets
const areaData = {
  labels: labels,
  datasets: [{
    label: 'Area (px²)',
    data: areaDataArr,
    borderColor: 'rgb(75, 192, 192)',
    tension: 0.3
  }]
};
const perimeterData = {
  labels: labels,
  datasets: [{
    label: 'Perimeter (px)',
    data: perimeterDataArr,
    borderColor: 'rgb(255, 99, 132)',
    tension: 0.3
  }]
};
const eccentricityData = {
  labels: labels,
  datasets: [{
    label: 'Eccentricity',
    data: eccentricityArr,
    borderColor: 'rgb(255, 206, 86)',
    tension: 0.3
  }]
};
const solidityData = {
  labels: labels,
  datasets: [{
    label: 'Solidity',
    data: solidityArr,
    borderColor: 'rgb(153, 102, 255)',
    tension: 0.3
  }]
};

// Store chart instances
const areaChartInstance = new Chart(document.getElementById('areaChart'), {
  type: 'line',
  data: areaData,
  options: { responsive: true }
});
const perimeterChartInstance = new Chart(document.getElementById('perimeterChart'), {
  type: 'line',
  data: perimeterData,
  options: { responsive: true }
});
const eccentricityChartInstance = new Chart(document.getElementById('eccentricityChart'), {
  type: 'line',
  data: eccentricityData,
  options: { responsive: true }
});
const solidityChartInstance = new Chart(document.getElementById('solidityChart'), {
  type: 'line',
  data: solidityData,
  options: { responsive: true }
});

// Handle window resize to force chart resize
window.addEventListener('resize', function() {
  areaChartInstance.resize();
  perimeterChartInstance.resize();
  eccentricityChartInstance.resize();
  solidityChartInstance.resize();
});

// CSV Download
window.downloadCSV = function() {
  let csv = '';
  const visibleCols = getVisibleCols();
  const headerCells = document.querySelectorAll('#report-table thead th');
  let header = [];
  headerCells.forEach(th => {
    if (visibleCols.includes(th.getAttribute('data-col'))) {
      header.push('"' + th.innerText + '"');
    }
  });
  csv += header.join(',') + '\n';

  document.querySelectorAll('#report-table tbody tr').forEach(row => {
    let rowData = [];
    visibleCols.forEach(col => {
      const cell = row.querySelector('.col-' + col);
      if (cell) rowData.push('"' + cell.innerText + '"');
    });
    csv += rowData.join(',') + '\n';
  });

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'report.csv';
  a.click();
  window.URL.revokeObjectURL(url);
};

window.downloadExcel = function() {
  const visibleCols = getVisibleCols();
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const trHead = document.createElement('tr');
  document.querySelectorAll('#report-table thead th').forEach(th => {
    if (visibleCols.includes(th.getAttribute('data-col'))) {
      const thClone = th.cloneNode(true);
      trHead.appendChild(thClone);
    }
  });
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  document.querySelectorAll('#report-table tbody tr').forEach(row => {
    const tr = document.createElement('tr');
    visibleCols.forEach(col => {
      const cell = row.querySelector('.col-' + col);
      if (cell) tr.appendChild(cell.cloneNode(true));
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  const wb = XLSX.utils.table_to_book(table, {sheet:"Report"});
  XLSX.writeFile(wb, "report.xlsx");
};

window.downloadPDF = function() {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: "landscape", unit: "mm", format: "a4" });

  doc.setFontSize(16);
  doc.text("Morphological Features Report", 10, 12);

  // Get visible graphs in the same order as on screen
  const chartOrder = [
    { id: 'areaChart', label: 'Area (px²)' },
    { id: 'perimeterChart', label: 'Perimeter (px)' },
    { id: 'eccentricityChart', label: 'Eccentricity' },
    { id: 'solidityChart', label: 'Solidity' }
  ];
  const visibleGraphs = getVisibleGraphs();

  // Filter and keep order
  const visibleCharts = chartOrder.filter(chart => visibleGraphs.includes(chart.id));

  // Layout: 2 columns per row
  const chartWidth = 120;
  const chartHeight = 55;
  const startX = 10;
  const startY = 18;
  const gapX = 10;
  const gapY = 10;
  const chartsPerRow = 2;

  let tableStartY = startY;
  if (visibleCharts.length > 0) {
    doc.setFontSize(12);
    visibleCharts.forEach(function(chart, idx) {
      const canvas = document.getElementById(chart.id);
      if (canvas) {
        // Calculate position
        const col = idx % chartsPerRow;
        const row = Math.floor(idx / chartsPerRow);
        const x = startX + col * (chartWidth + gapX);
        const y = startY + row * (chartHeight + gapY + 8); // 8 for label space

        doc.text(chart.label, x, y - 2);
        const imgData = canvas.toDataURL("image/png", 1.0);
        doc.addImage(imgData, 'PNG', x, y, chartWidth, chartHeight);

        // Update tableStartY to be after the last chart
        if (idx === visibleCharts.length - 1) {
          tableStartY = y + chartHeight + gapY;
        }
      }
    });
    // Table on a new page if there are graphs
    doc.addPage();
    doc.setFontSize(14);
    doc.text("Data Table", 10, 15);
    tableStartY = 20;
  } else {
    // Table on the first page if no graphs
    doc.setFontSize(14);
    doc.text("Data Table", 10, 18);
    tableStartY = 23;
  }

  // Build table data for visible columns
  const visibleCols = getVisibleCols();
  const head = [];
  document.querySelectorAll('#report-table thead th').forEach(th => {
    if (visibleCols.includes(th.getAttribute('data-col'))) {
      head.push(th.innerText);
    }
  });
  const body = [];
  document.querySelectorAll('#report-table tbody tr').forEach(row => {
    const rowData = [];
    visibleCols.forEach(col => {
      const cell = row.querySelector('.col-' + col);
      if (cell) rowData.push(cell.innerText);
    });
    body.push(rowData);
  });

  doc.autoTable({
    head: [head],
    body: body,
    startY: tableStartY,
    theme: 'grid',
    headStyles: { fillColor: [75, 192, 192] },
    styles: { fontSize: 9 }
  });

  doc.save("report.pdf");
};

// Column visibility toggles
document.querySelectorAll('.col-toggle').forEach(function(checkbox) {
  checkbox.addEventListener('change', function() {
    const col = this.value;
    const show = this.checked;
    document.querySelectorAll('.col-' + col).forEach(function(cell) {
      cell.style.display = show ? '' : 'none';
    });
  });
});

// Graph visibility toggles with dynamic placement
document.querySelectorAll('.graph-toggle').forEach(function(checkbox) {
  checkbox.addEventListener('change', updateChartGrid);
});

// On page load, apply initial graph visibility and placement
window.addEventListener('DOMContentLoaded', updateChartGrid);

function updateChartGrid() {
  const chartOrder = [
    'areaChart-wrapper',
    'perimeterChart-wrapper',
    'eccentricityChart-wrapper',
    'solidityChart-wrapper'
  ];
  const chartsRow = document.getElementById('charts-row');
  // Get checked graphs in order
  const checked = Array.from(document.querySelectorAll('.graph-toggle'))
    .filter(cb => cb.checked)
    .map(cb => cb.value + '-wrapper');
  // Show and append checked wrappers in order
  checked.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.style.display = '';
      chartsRow.appendChild(el);
    }
  });
  // Hide unchecked wrappers
  chartOrder.forEach(id => {
    if (!checked.includes(id)) {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    }
  });
  // Force all charts to resize
  areaChartInstance.resize();
  perimeterChartInstance.resize();
  eccentricityChartInstance.resize();
  solidityChartInstance.resize();
}

// Helper to get visible columns
function getVisibleCols() {
  return Array.from(document.querySelectorAll('.col-toggle'))
    .filter(cb => cb.checked)
    .map(cb => cb.value);
}

// Helper to get visible graphs
function getVisibleGraphs() {
  return Array.from(document.querySelectorAll('.graph-toggle'))
    .filter(cb => cb.checked)
    .map(cb => cb.value);
}