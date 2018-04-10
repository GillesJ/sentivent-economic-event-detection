function item_showhide(showhide, id) {
    var myid = id.id;

    // the typical id is showhide_SOMETHING_REALID and we want REALID
    var i = 0;
    while(myid != '' && i != 2) {
	if (myid.charAt(0) == '_') {
	    i++;
	}
	myid = myid.substr(1);
    }
  
    

    if (showhide == 'show') {
	var e_plus = document.getElementById("showhide_plus_"+myid);
	e_plus.style.display = 'none';

	var e_minus = document.getElementById("showhide_minus_"+myid);
	e_minus.style.display = 'block';

	var e_short = document.getElementById("showhide_short_"+myid);
	e_short.style.display = 'none';

	var e_details = document.getElementById("showhide_details_"+myid);
	e_details.style.display = 'block';
    }

    if (showhide == 'hide') {
	var e_plus = document.getElementById("showhide_plus_"+myid);
	e_plus.style.display = 'block';

	var e_minus = document.getElementById("showhide_minus_"+myid);
	e_minus.style.display = 'none';

	var e_short = document.getElementById("showhide_short_"+myid);
	e_short.style.display = 'block';

	var e_details = document.getElementById("showhide_details_"+myid);
	e_details.style.display = 'none';
    }
}


function toggle_visibility(id) {
  var e = document.getElementById(id);
  if(e.style.display == 'none')
    e.style.display = 'block';
  else
    e.style.display = 'none';
}

function toggle_visibility_dialog(id) {
  jQuery(function() {
    if (jQuery( "#"+id ).dialog("isOpen") == true) {
      jQuery( "#"+id ).dialog("close");
    } else {
      jQuery( "#"+id ).dialog("open");
    }
  });
}

function toggle_visibility_sticky(id,name) {
  var e = document.getElementById(id);
  if(e.style.display == 'block') {
    e.style.display = 'none';
    jQuery.ajax({
	    url: 'scmd.cgi?scmd=stickyshowhide&code='+name+'&hide=0&mytime='+new Date().getTime()});
  } else {
    e.style.display = 'block';
    jQuery.ajax({
	    url: 'scmd.cgi?scmd=stickyshowhide&code='+name+'&hide=1&mytime='+new Date().getTime()});
  }
}

function toggle_visibility_dialog_sticky(id,name) {
  jQuery(function() {
    if (jQuery( "#"+id ).dialog("isOpen") == true) {
      jQuery( "#"+id ).dialog("close");
      jQuery.ajax({
	      url: 'scmd.cgi?scmd=stickyshowhide&code='+name+'&hide=0&mytime='+new Date().getTime()});
    } else {
      jQuery( "#"+id ).dialog("open");
      jQuery.ajax({
	      url: 'scmd.cgi?scmd=stickyshowhide&code='+name+'&hide=1&mytime='+new Date().getTime()});
    }
  });
}


function toggle_visibility_double(id) {
  toggle_visibility(id);
  toggle_visibility('hidden_'+id);
}

function StickyTableUpdate(id, size) {
      jQuery.ajax({
	      url: 'scmd.cgi?scmd=stickytable&code='+id+'&pagesize='+size+'&mytime='+new Date().getTime()}); 
}

// used to check numeric fields
// to be applied using onkeyup=numeric_field_change(this)
function numeric_field_change(field) {
    var check = true;
    var value = field.value; //get characters
    //check that all characters are digits, ., -, or ""
    for(var i=0;i < field.value.length; ++i)
	{
	    var new_key = value.charAt(i); //cycle through characters
	    if(((new_key < "0") || (new_key > "9")) && 
	       !(new_key == "") &&
	       !(new_key == ".")
	       )
		{
                    check = false;
                    break;
		}
	}
    //apply appropriate colour based on value
    if(!check)
	{
	    field.style.backgroundColor = "red";
	}
    else
	{
	    field.style.backgroundColor = "white";
	}
}

// used to check numeric fields
// to be applied using onkeyup=numeric_field_change(this)
function integer_field_change(field) {
    var check = true;
    var value = field.value; //get characters
    //check that all characters are digits, ., -, or ""
    for(var i=0;i < field.value.length; ++i)
	{
	    var new_key = value.charAt(i); //cycle through characters
	    if(((new_key < "0") || (new_key > "9")) && 
	       !(new_key == "") &&
	       !(new_key == ".") &&
	       !((new_key == "-") && (i==0))
	       )
		{
                    check = false;
                    break;
		}
	}
    //apply appropriate colour based on value
    if(!check)
	{
	    field.style.backgroundColor = "red";
	}
    else
	{
	    field.style.backgroundColor = "white";
	}
}


// used to check the length of labels
// to be applied using onkeyup=label_field_change(this)
function label_field_change(field) {
    var check = true;
    var value = field.value;

    var box = document.getElementById("box_"+field.name);

    if (field.value.length < 20) {
	// everything ok
	field.style.backgroundColor = "white";
	box.innerHTML = "";
	box.style.backgroundColor = "white";
    } else if (field.value.length < 40) {
	// a little bit too long!
	field.style.backgroundColor = "orange";
	box.innerHTML = "Please keep it short!<br>(you can use a Text item before the current one to provide a description...)";
	box.style.backgroundColor = "orange";
    } else {
	// way too long!
	field.style.backgroundColor = "red";
	box.innerHTML = "Please keep it short!<br>(you can use a Text item before the current one to provide a description...)<br>WARNING: the content may not align properly when displayed or sent by mail!";
	box.style.backgroundColor = "red";
    }
}

// From stackoverflow.com
function insertTextAtCursor(el, text) {
    var val = el.value, endIndex, range;
    if (typeof el.selectionStart != "undefined" && typeof el.selectionEnd != "undefined") {
        endIndex = el.selectionEnd;
        el.value = val.slice(0, el.selectionStart) + text + val.slice(endIndex);
        el.selectionStart = el.selectionEnd = endIndex + text.length;
    } else if (typeof document.selection != "undefined" && typeof document.selection.createRange != "undefined") {
        el.focus();
        range = document.selection.createRange();
        range.collapse(false);
        range.text = text;
        range.select();
    }
}
