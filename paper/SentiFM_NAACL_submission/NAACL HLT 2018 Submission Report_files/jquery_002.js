/**
 * Copyright 2010 Ganzogo. All Rights Reserved.
 *
 * jQuery plugin that adds buttons to text input fields which allow the user
 * to easily enter special characters.
 *
 * Author: Matthew Hasler (matthew@ganzogo.com)
 */
(function($) {
  $.fn.specialedit = function(chars, options) {

    var KEYCODE_SHIFT = 16;
    var ESZETT = '&#223;';

    var settings = {
      toolbarBgColor: '#f0f0ee',
      toolbarBorderColor: '#cccccc',
      buttonBgColor: '#b2bbd0',
      buttonBorderColor: '#000000',
      buttonTextColor: '#000000',
      buttonWidth: 20,
      buttonHeight: 20,
      buttonMargin: 3,
      lineNumber: 1
    };

    if (options) {
      $.extend(settings, options);
    }

    return this.each(function() {

      // Create the toolbar for this text input.
      var $this = $(this);
      var hover = false;
      var buttons = new Array();
      var button_top = $this.offset().top + $this.outerHeight() + 5;
      var button_right = $this.offset().left + $this.outerWidth() - 2;

      var linelength = Math.ceil(chars.length / settings.lineNumber);
      var lineheight = settings.buttonHeight + 2 * (settings.buttonMargin + 1);

      var toolbar = $('<div/>').css('position', 'absolute')
          .css('left', (button_right - (linelength *
              (settings.buttonWidth + settings.buttonMargin + 2)) -
              settings.buttonMargin) + 'px')
          .css('top', button_top + 'px')
          .css('display', 'none')
          .css('border', '1px solid ' + settings.toolbarBorderColor)
          .css('width', ((linelength *
              (settings.buttonWidth + settings.buttonMargin + 2)) +
              settings.buttonMargin) + 'px')
          .css('height', (lineheight*settings.lineNumber) + 'px')
          .css('background', settings.toolbarBgColor)
          .mouseover(function() {
              hover = true;
          })
          .mouseout(function() {
              hover = false;
          })
          .click(function() {
              $this.focus();
          });

      // Create each of the buttons on the toolbar.
      $.each(chars, function(i, c) {
        buttons[i] = $('<div/>').html(c)
            .css('width', settings.buttonWidth + 'px')
            .css('height', (settings.buttonHeight -
                (settings.buttonHeight / 2) + 9) + 'px')
            .css('color', (settings.buttonTextColor))
            .css('border', '0')
            .css('text-align', 'center')
            .css('cursor', 'pointer')
            .css('top', (settings.buttonMargin + lineheight*(Math.floor(i/linelength)) ) + 'px')
            .css('left', (((settings.buttonWidth + settings.buttonMargin + 2) *
                (i%linelength)) + settings.buttonMargin)  + 'px')
            .css('border', '1px solid ' + settings.toolbarBgColor)
            .css('padding-top', ((settings.buttonHeight / 2) - 8) + 'px')
            .css('position', 'absolute')
            .mouseover(function() {
                $(this).css('border', '1px solid ' +
                    settings.buttonBorderColor)
                       .css('background', settings.buttonBgColor);
            })
            .mouseout(function() {
                $(this).css('border', '1px solid ' + settings.toolbarBgColor)
                       .css('background', settings.toolbarBgColor);
            })
            .mousedown(function() {
                var input = $this[0];
                var value = htmlDecode($(this).html());
                if (document.selection) {
                  input.focus();
                  sel = document.selection.createRange();
                  sel.text = value;
                  input.focus();
                } else if (input.selectionStart ||
                    input.selectionStart == '0') {
                  var startPos = input.selectionStart;
                  var endPos = input.selectionEnd;
                  var scrollTop = input.scrollTop;
                  input.value = input.value.substring(0, startPos) + value +
                      input.value.substring(endPos, input.value.length);
                  input.focus();
                  input.selectionStart = startPos + value.length;
                  input.selectionEnd = startPos + value.length;
                  input.scrollTop = scrollTop;
                } else {
                  input.value += value;
                  input.focus();
                }
            });
        toolbar.append(buttons[i]);
      });

      $this.after(toolbar);

      function htmlDecode(value) {
        return $('<div/>').html(value).text();
      }

      // Bind events to text input using '.specialedit' namespace.
      $this.bind('keydown.specialedit', function(event) {
        if (event.which == KEYCODE_SHIFT) {
          for (var c in buttons) {
            // The eszett doesn't play well with toUpperCase method.
            if (buttons[c].html() != htmlDecode(ESZETT)) {
              buttons[c].html(buttons[c].html().toUpperCase());
            }
          }
        }
      });

      $this.bind('keyup.specialedit', function(event) {
        if (event.which == KEYCODE_SHIFT) {
          for (var c in buttons) {
            buttons[c].html(buttons[c].html().toLowerCase());
          }
        }
      });

      $this.bind('focus.specialedit', function() {
        toolbar.fadeIn();
      });

      /** 
       * We check to see if the mouse is currently over the buttons. If it is
       * then we take this to mean that the user has clicked a button, and
       * that's why the blur event has been triggered. In that case, the
       * buttons shouldn't be hidden.
       * 
       * This causes a minor bug if the user is hovering over the buttons,
       * while the text field loses focus some over way, (for example, by
       * pressing TAB).
       */
      $this.bind('blur.specialedit', function() {
        if (!hover) {
          toolbar.hide();
        }
      });
    });
  };
})( jQuery );
