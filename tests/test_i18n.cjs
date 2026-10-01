const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function fixture() {
  const controls = new Map();
  const button = id => {
    if (!controls.has(id)) controls.set(id, {
      attributes: {}, events: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(name, handler) { this.events[name] = handler; },
    });
    return controls.get(id);
  };
  const meta = {content:''};
  const context = vm.createContext({
    window: {}, localStorage: {getItem:()=>null,setItem(){}},
    document: {
      body: {}, documentElement: {lang:'en'}, title:'',
      getElementById:button,
      createTreeWalker:()=>({nextNode:()=>false}),
      querySelectorAll:()=>[], querySelector:()=>meta,
    },
    NodeFilter: {SHOW_TEXT:4},
    MutationObserver: class {observe(){} disconnect(){}},
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../app/i18n.js'),'utf8'), context);
  return {i18n:context.window.sign2textI18n, button, context};
}

test('Russian mode translates controls, results, errors, and finger definitions', () => {
  const {i18n, context} = fixture();
  assert.equal(context.document.documentElement.lang,'ru');
  assert.equal(i18n.translate('Enable camera'),'Включить камеру');
  assert.equal(i18n.translate('Stop + Yes'),'Стоп + Да');
  assert.equal(i18n.translate('Two fingers: thumb + middle'),'Два пальца: большой + средний');
  assert.equal(i18n.translate('Gesture not recognized. Hold one or two supported gestures clearly for a second.'),
    'Жест не распознан. Чётко покажите один или два жеста около секунды.');
  assert.equal(i18n.translate('1.2s'),'1.2 с');
  assert.equal(i18n.translate('CAT'),'CAT');
});

test('language switch restores English and can switch back to Russian', () => {
  const {i18n, button, context} = fixture();
  i18n.setLocale('en');
  assert.equal(context.document.documentElement.lang,'en');
  assert.equal(i18n.translate('Yes'),'Yes');
  assert.equal(button('localeEn').attributes['aria-pressed'],'true');
  i18n.setLocale('ru');
  assert.equal(i18n.translate('Yes'),'Да');
  assert.equal(button('localeRu').attributes['aria-pressed'],'true');
});
