import React, { useState, useEffect } from 'react';

const SUPABASE_URL = 'https://jjnscpkrmxpiigmtirvp.supabase.co';
const SUPABASE_ANON_KEY = 'YOUR_ANON_KEY_HERE'; // 실제 키로 교체 필요

export default function PassageExtractor() {
  const [schoolName, setSchoolName] = useState('');
  const [passages, setPassages] = useState([]);
  const [books, setBooks] = useState({});
  const [selected, setSelected] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [expandedBooks, setExpandedBooks] = useState(new Set());

  // Supabase에서 지문 데이터 가져오기
  useEffect(() => {
    async function fetchPassages() {
      try {
        const res = await fetch(
          `${SUPABASE_URL}/rest/v1/passages?select=id,book,unit,pid,passage_text&order=book,unit,pid`,
          {
            headers: {
              apikey: SUPABASE_ANON_KEY,
              Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
            },
          }
        );
        const data = await res.json();
        setPassages(data);

        // 교재별로 그룹화
        const grouped = {};
        data.forEach((p) => {
          if (!grouped[p.book]) grouped[p.book] = {};
          if (!grouped[p.book][p.unit]) grouped[p.book][p.unit] = [];
          grouped[p.book][p.unit].push(p);
        });
        setBooks(grouped);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch passages:', err);
        setLoading(false);
      }
    }
    fetchPassages();
  }, []);

  // 체크박스 토글
  const toggleSelect = (id) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // 교재 전체 선택/해제
  const toggleBook = (book) => {
    const bookPassages = Object.values(books[book] || {}).flat();
    const allSelected = bookPassages.every((p) => selected.has(p.id));
    setSelected((prev) => {
      const next = new Set(prev);
      bookPassages.forEach((p) => {
        if (allSelected) next.delete(p.id);
        else next.add(p.id);
      });
      return next;
    });
  };

  // 단원 전체 선택/해제
  const toggleUnit = (book, unit) => {
    const unitPassages = books[book]?.[unit] || [];
    const allSelected = unitPassages.every((p) => selected.has(p.id));
    setSelected((prev) => {
      const next = new Set(prev);
      unitPassages.forEach((p) => {
        if (allSelected) next.delete(p.id);
        else next.add(p.id);
      });
      return next;
    });
  };

  // 교재 접기/펼치기
  const toggleExpand = (book) => {
    setExpandedBooks((prev) => {
      const next = new Set(prev);
      if (next.has(book)) next.delete(book);
      else next.add(book);
      return next;
    });
  };

  // HTML 생성 및 새 창에서 열기
  const extractHTML = () => {
    if (selected.size === 0) {
      alert('지문을 선택해주세요!');
      return;
    }

    const school = schoolName || '학교명';
    const header = `${school}/정독하며 모르는 단어 체크/ 기억 안나는 지문 체크/ 체크한 곳 리뷰/ 그리고는 3번 정독 (1회), (2회), (3회)`;

    // 선택된 지문들 가져오기 (book, unit, pid 순서로 정렬)
    const selectedPassages = passages
      .filter((p) => selected.has(p.id))
      .sort((a, b) => {
        if (a.book !== b.book) return a.book.localeCompare(b.book);
        if (a.unit !== b.unit) return a.unit.localeCompare(b.unit);
        return a.pid.localeCompare(b.pid);
      });

    let content = '';
    selectedPassages.forEach((p) => {
      const label = `${p.book} ${p.unit} ${p.pid}`;
      content += `<div style="margin-bottom: 28px; page-break-inside: avoid;">
  <p style="font-weight: bold; color: #333; margin-bottom: 10px; font-size: 11pt;">${label}</p>
  <p style="line-height: 2.0; text-align: justify; font-size: 11pt;">${p.passage_text}</p>
</div>\n`;
    });

    const html = `<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>원문 정독 - ${school}</title>
<style>
@page { size: A4; margin: 15mm 18mm; }
@media print {
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
body { 
  font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; 
  font-size: 11pt; 
  line-height: 1.6; 
  color: #000; 
  max-width: 210mm;
  margin: 0 auto;
  padding: 20px;
}
.header { 
  border-bottom: 2px solid #333; 
  padding-bottom: 10px; 
  margin-bottom: 24px; 
  font-size: 10pt; 
  color: #333;
  font-weight: 500;
}
</style>
</head>
<body>
<div class="header">${header}</div>
${content}
</body>
</html>`;

    const newWindow = window.open('', '_blank');
    newWindow.document.write(html);
    newWindow.document.close();
  };

  // 선택 초기화
  const clearSelection = () => setSelected(new Set());

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">지문 데이터 불러오는 중...</div>
      </div>
    );
  }

  const bookNames = Object.keys(books).sort();

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h2 className="text-xl font-bold text-blue-600 mb-4 border-b-2 border-orange-400 pb-2 inline-block">
        📖 원문 추출
      </h2>

      {/* 학교명 입력 */}
      <div className="mb-6">
        <label className="block text-sm text-gray-600 mb-2">학교 이름</label>
        <input
          type="text"
          value={schoolName}
          onChange={(e) => setSchoolName(e.target.value)}
          placeholder="예: 정명고2"
          className="border border-gray-300 rounded px-3 py-2 w-64 focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* 선택 현황 */}
      <div className="mb-4 flex items-center gap-4">
        <span className="text-sm text-gray-600">
          선택된 지문: <strong className="text-blue-600">{selected.size}개</strong>
        </span>
        <button
          onClick={clearSelection}
          className="text-sm text-red-500 hover:underline"
        >
          선택 초기화
        </button>
      </div>

      {/* 교재 목록 */}
      <div className="border rounded-lg overflow-hidden mb-6 max-h-96 overflow-y-auto">
        {bookNames.map((book) => {
          const units = Object.keys(books[book]).sort();
          const bookPassages = Object.values(books[book]).flat();
          const selectedCount = bookPassages.filter((p) => selected.has(p.id)).length;
          const isExpanded = expandedBooks.has(book);

          return (
            <div key={book} className="border-b last:border-b-0">
              {/* 교재 헤더 */}
              <div
                className="flex items-center gap-2 p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
                onClick={() => toggleExpand(book)}
              >
                <span className="text-gray-400">{isExpanded ? '▼' : '▶'}</span>
                <input
                  type="checkbox"
                  checked={selectedCount === bookPassages.length && bookPassages.length > 0}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleBook(book);
                  }}
                  className="w-4 h-4"
                />
                <span className="font-medium text-sm flex-1">{book}</span>
                <span className="text-xs text-gray-500">
                  {selectedCount}/{bookPassages.length}
                </span>
              </div>

              {/* 단원 목록 (펼쳐진 경우) */}
              {isExpanded && (
                <div className="pl-8 py-2 bg-white">
                  {units.map((unit) => {
                    const unitPassages = books[book][unit];
                    const unitSelected = unitPassages.filter((p) => selected.has(p.id)).length;

                    return (
                      <div key={unit} className="mb-2">
                        {/* 단원 헤더 */}
                        <div className="flex items-center gap-2 mb-1">
                          <input
                            type="checkbox"
                            checked={unitSelected === unitPassages.length}
                            onChange={() => toggleUnit(book, unit)}
                            className="w-3.5 h-3.5"
                          />
                          <span className="text-sm font-medium text-blue-600">{unit}</span>
                          <span className="text-xs text-gray-400">
                            ({unitSelected}/{unitPassages.length})
                          </span>
                        </div>

                        {/* 지문 목록 */}
                        <div className="pl-6 flex flex-wrap gap-2">
                          {unitPassages.map((p) => (
                            <label
                              key={p.id}
                              className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs cursor-pointer transition-colors ${
                                selected.has(p.id)
                                  ? 'bg-blue-100 text-blue-700 border border-blue-300'
                                  : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={selected.has(p.id)}
                                onChange={() => toggleSelect(p.id)}
                                className="w-3 h-3"
                              />
                              {p.pid}
                            </label>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 버튼 */}
      <div className="flex gap-3">
        <button
          onClick={extractHTML}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
        >
          HTML 추출 (새 창)
        </button>
        <span className="text-sm text-gray-500 self-center">
          → 새 창에서 Ctrl+P로 PDF 저장
        </span>
      </div>
    </div>
  );
}
