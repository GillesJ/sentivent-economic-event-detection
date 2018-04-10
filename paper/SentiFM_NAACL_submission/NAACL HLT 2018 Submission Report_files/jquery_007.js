function setToolTip() {
    jQuery( document ).tooltip({
      items: "[tagtitle], [title]",
      content: function() {
        var element = jQuery( this );
        if ( element.is( "[tagtitle]" ) ) {
          return jQuery("#" + element.attr( "tagtitle" )).html();
        }
        if ( element.is( "[title]" ) ) {
          return element.attr( "title" );
        }
      }
    });
}


// Active tab
function setTabMagic() {
    //  To return to the same tab on refresh and redirect.
    //
    //  Documentation
    //      http://api.jqueryui.com/tabs/#option-active
    //      http://api.jqueryui.com/tabs/#event-activate
    //      http://balaarjunan.wordpress.com/2010/11/10/html5-session-storage-key-things-to-consider/
    //
    //  Define friendly index name
    var index = 'key';
    //  Define friendly data store name
    var dataStore = window.sessionStorage;
    //  Start magic!
    try {
        // getter: Fetch previous value
        var oldIndex = dataStore.getItem(index);
    } catch(e) {
        // getter: Always default to first tab in error state
        var oldIndex = 0;
    }
    jQuery('#tabs').tabs({
        // The zero-based index of the panel that is active (open)
        active : oldIndex,
        // Triggered after a tab has been activated
        activate : function( event, ui ){
            //  Get future value
            var newIndex = ui.newTab.parent().children().index(ui.newTab);
            //  Set future value
            dataStore.setItem( index, newIndex ) 
        }
    });

    var hash = jQuery.trim( window.location.hash );
    if (hash) {
      var index = jQuery('#tabs a[href="'+hash+'"]').parent().index();
      jQuery("#tabs").tabs("option", "active", index);
    }
}
